import { ApiError, isErrorBody, readErrorResponse, request, trimBaseUrl } from '../client.js'
import { readServerSentEvents } from '../sse.js'
import type {
  FileBlockParam,
  MessageParam,
  MessageStreamEvent,
  MessageUsage,
  StopReason,
  ThinkingLevel,
} from '../types.js'
import { BlockStream, mapFinishReason, usageOf } from './blocks.js'
import { localRef } from './inline.js'
import { asString, isRecord } from './json.js'
import type { SkillSource } from './skills.js'
import {
  mergeSkillSources,
  readSearchQueries,
  readSkillSources,
  searchToolResult,
  searchToolUse,
  sourceCitation,
  uniqueSkillSources,
} from './skills.js'
import type { AttachmentLoader, ProtocolAdapter, StreamOptions } from './types.js'
import { UNAVAILABLE_ATTACHMENT_TEXT } from './types.js'

const baseUrl = (base: string) => trimBaseUrl(base)

/** 模型名可能带 models/ 前缀（从 /models 拉取时就是如此），拼 URL 前去掉 */
const modelPath = (model: string) => model.replace(/^models\//, '')

/** 思考等级映射到 Gemini 的 thinkingBudget（token 数） */
const THINKING_BUDGET: Record<ThinkingLevel, number> = {
  minimal: 1024,
  low: 2048,
  medium: 4096,
  high: 8192,
  xhigh: 16384,
  max: 24576,
}

const headers = (apiKey: string) => ({
  'x-goog-api-key': apiKey,
  'Content-Type': 'application/json',
})

const textOf = (content: Extract<MessageParam, { role: 'assistant' }>['content']) =>
  content
    .flatMap((block) => (block.type === 'text' ? [block.text] : []))
    .join('')
    .trim()

/** Gemini 的 parts：文本直接给，图片与 PDF 用 inlineData，其余类型跳过 */
const filePart = async (block: FileBlockParam, load: AttachmentLoader) => {
  const ref = localRef(block)
  const loaded = ref ? await load(ref) : null
  return loaded ? { inlineData: { mimeType: loaded.mediaType, data: loaded.base64 } } : null
}

const userParts = async (
  content: string | (FileBlockParam | { type: 'text'; text: string })[],
  load: AttachmentLoader
) => {
  if (typeof content === 'string') {
    return content ? [{ text: content }] : []
  }
  const parts: Record<string, unknown>[] = []
  for (const block of content) {
    if (block.type === 'text') {
      if (block.text) {
        parts.push({ text: block.text })
      }
      continue
    }
    const part = await filePart(block, load)
    if (part) {
      parts.push(part)
    }
  }
  return parts
}

/**
 * 最近一次上下文快照在 Gemini 里没有独立的 system 角色，
 * 作为下一轮用户消息的第一段文字，保留时间与位置信息。
 */
const toContents = async (messages: MessageParam[], load: AttachmentLoader) => {
  const contents: Record<string, unknown>[] = []
  let reminder = ''
  for (const message of messages) {
    if (message.role === 'latest_reminder') {
      reminder = message.content
      continue
    }
    if (message.role === 'user') {
      const parts = await userParts(message.content, load)
      if (parts.length === 0) {
        parts.push({ text: UNAVAILABLE_ATTACHMENT_TEXT })
      }
      if (reminder) {
        parts.unshift({ text: reminder })
        reminder = ''
      }
      contents.push({ role: 'user', parts })
      continue
    }
    const text = textOf(message.content)
    if (text) {
      contents.push({ role: 'model', parts: [{ text }] })
    }
  }
  return contents
}

const readUsage = (value: unknown): MessageUsage | undefined => {
  if (!isRecord(value)) {
    return undefined
  }
  const input = typeof value.promptTokenCount === 'number' ? value.promptTokenCount : undefined
  const output =
    typeof value.candidatesTokenCount === 'number' ? value.candidatesTokenCount : undefined
  if (input === undefined && output === undefined) {
    return undefined
  }
  const cached =
    typeof value.cachedContentTokenCount === 'number' ? value.cachedContentTokenCount : 0
  return usageOf(input ?? 0, output ?? 0, cached)
}

async function* stream(options: StreamOptions): AsyncGenerator<MessageStreamEvent> {
  const { provider, deepseek, request: chat, loadAttachment, signal } = options
  const contents = await toContents(chat.messages, loadAttachment)
  const body: Record<string, unknown> = {
    systemInstruction: { parts: [{ text: chat.system }] },
    contents,
  }
  if (chat.thinking) {
    body.generationConfig = {
      thinkingConfig: {
        includeThoughts: true,
        thinkingBudget: THINKING_BUDGET[chat.thinkingLevel],
      },
    }
  }
  if (!deepseek && chat.search) {
    // Gemini 的标准内置搜索；结果通过候选上的 groundingMetadata 返回。
    body.tools = [{ googleSearch: {} }]
  }
  const url = `${baseUrl(provider.baseUrl)}/models/${encodeURIComponent(modelPath(chat.model))}:streamGenerateContent?alt=sse`
  const response = await request(url, {
    method: 'POST',
    headers: { ...headers(provider.apiKey), Accept: 'text/event-stream' },
    body: JSON.stringify(body),
    signal,
  })
  const contentType = response.headers.get('Content-Type') ?? ''
  if (!response.body || !contentType.includes('text/event-stream')) {
    throw await readErrorResponse(response, 'protocol', '')
  }

  const blocks = new BlockStream()
  let usage: MessageUsage | undefined
  // groundingMetadata 可能只在最后一个分片给出；先累计，收尾时再决定展示方式。
  const groundingSources: SkillSource[] = []
  const groundingQueries: string[] = []
  let searchId = ''
  let searchEmitted = false
  let textStarted = false
  for await (const { data } of readServerSentEvents(response.body, signal)) {
    let payload: unknown
    try {
      payload = JSON.parse(data)
    } catch {
      continue
    }
    if (isErrorBody(payload)) {
      throw new ApiError('stream', payload.error.message)
    }
    if (!isRecord(payload)) {
      continue
    }
    usage = readUsage(payload.usageMetadata) ?? usage
    const blocked = asString(
      (isRecord(payload.promptFeedback) ? payload.promptFeedback : {}).blockReason
    )
    if (blocked) {
      // 输入被安全策略拦截：没有正文，按拒绝收尾让界面展示统一提示。
      yield* blocks.finish('refusal', usage)
      return
    }
    const candidate = Array.isArray(payload.candidates) ? payload.candidates[0] : undefined
    if (!isRecord(candidate)) {
      continue
    }
    // 少数代理会在首个分片就带 grounding；能确定来源时先声明搜索步骤，结果在收尾补齐。
    const grounding =
      !deepseek && isRecord(candidate.groundingMetadata) ? candidate.groundingMetadata : undefined
    if (grounding) {
      mergeSkillSources(groundingSources, readSkillSources(grounding.groundingChunks))
      for (const query of readSearchQueries(grounding.webSearchQueries)) {
        if (!groundingQueries.includes(query)) {
          groundingQueries.push(query)
        }
      }
      if (!searchEmitted && !textStarted && groundingSources.length > 0) {
        searchId = `gemini_web_${Date.now()}`
        searchEmitted = true
        yield* blocks.tool(searchToolUse(searchId, groundingQueries))
      }
    }
    const content = isRecord(candidate.content) ? candidate.content : undefined
    const parts = content && Array.isArray(content.parts) ? content.parts : []
    for (const part of parts) {
      if (!isRecord(part)) {
        continue
      }
      const value = asString(part.text)
      if (part.thought === true) {
        yield* blocks.thinking(value)
      } else {
        if (value) {
          textStarted = true
        }
        yield* blocks.text(value)
      }
    }
    const stopReason: StopReason | null = mapFinishReason(asString(candidate.finishReason))
    if (stopReason) {
      if (searchEmitted) {
        yield* blocks.tool(searchToolResult(searchId, groundingSources))
      } else if (groundingSources.length > 0) {
        // Gemini 通常在正文之后才给 grounding，无法回头插入调用块；
        // 先补文本引用，再补一个没有调用块的来源结果，底部来源卡片仍能展示。
        for (const source of uniqueSkillSources(groundingSources)) {
          yield* blocks.citation(sourceCitation(source))
        }
        yield* blocks.tool(searchToolResult(`gemini_grounding_${Date.now()}`, groundingSources))
      }
      yield* blocks.finish(stopReason, usage)
      return
    }
  }
  throw new ApiError('interrupted', '')
}

const modelIds = (body: unknown): string[] => {
  const models = isRecord(body) ? body.models : undefined
  if (!Array.isArray(models)) {
    return []
  }
  return models.flatMap((item) => {
    if (!isRecord(item)) {
      return []
    }
    const name = asString(item.name)
    if (!name) {
      return []
    }
    const methods = Array.isArray(item.supportedGenerationMethods)
      ? item.supportedGenerationMethods.filter(
          (method): method is string => typeof method === 'string'
        )
      : []
    return methods.length === 0 || methods.includes('generateContent')
      ? [name.replace(/^models\//, '')]
      : []
  })
}

const listModels = async (provider: { baseUrl: string; apiKey: string }, signal?: AbortSignal) => {
  const response = await request(`${baseUrl(provider.baseUrl)}/models?pageSize=200`, {
    method: 'GET',
    headers: { 'x-goog-api-key': provider.apiKey },
    signal,
  })
  return modelIds(await response.json())
}

export const geminiAdapter: ProtocolAdapter = {
  id: 'gemini',
  stream,
  async verify(provider, signal) {
    await listModels(provider, signal)
  },
  listModels,
}
