import { ApiError, isErrorBody, readErrorResponse, request, trimBaseUrl } from '../client.js'
import { readServerSentEvents } from '../sse.js'
import type {
  FileBlockParam,
  MessageParam,
  MessageStreamEvent,
  MessageUsage,
  StopReason,
} from '../types.js'
import { BlockStream, mapFinishReason, usageOf } from './blocks.js'
import { dataUrl, localRef } from './inline.js'
import { asNumber, asString, at, isRecord } from './json.js'
import type { AttachmentLoader, ProtocolAdapter, StreamOptions } from './types.js'
import { UNAVAILABLE_ATTACHMENT_TEXT } from './types.js'

const baseUrl = (base: string) => trimBaseUrl(base)

const authorization = (apiKey: string) => ({
  Authorization: `Bearer ${apiKey}`,
  'Content-Type': 'application/json',
})

/** 文本增量；assistant 历史只回传正文，思考与工具内容对多数兼容服务无效 */
const textOf = (content: Extract<MessageParam, { role: 'assistant' }>['content']) =>
  content
    .flatMap((block) => (block.type === 'text' ? [block.text] : []))
    .join('')
    .trim()

const userParts = async (
  blocks: (FileBlockParam | { type: 'text'; text: string })[],
  load: AttachmentLoader
) => {
  const parts: Record<string, unknown>[] = []
  for (const block of blocks) {
    if (block.type === 'text') {
      if (block.text) {
        parts.push({ type: 'text', text: block.text })
      }
      continue
    }
    const ref = localRef(block)
    const loaded = ref ? await load(ref) : null
    // 只内联图片；PDF 在 Chat Completions 里没有通用写法，其余附件跳过。
    if (loaded && loaded.mediaType.startsWith('image/')) {
      parts.push({
        type: 'image_url',
        image_url: { url: dataUrl(loaded.mediaType, loaded.base64) },
      })
    }
  }
  return parts
}

const toUserContent = async (
  content: string | (FileBlockParam | { type: 'text'; text: string })[],
  load: AttachmentLoader
): Promise<string | Record<string, unknown>[]> => {
  if (typeof content === 'string') {
    return content
  }
  const parts = await userParts(content, load)
  if (parts.length === 0) {
    return UNAVAILABLE_ATTACHMENT_TEXT
  }
  // 只有文字时退回字符串，兼容只接受字符串的老服务。
  return parts.length === 1 && parts[0].type === 'text' ? String(parts[0].text) : parts
}

const toMessages = async (messages: MessageParam[], load: AttachmentLoader) => {
  const result: Record<string, unknown>[] = []
  for (const message of messages) {
    if (message.role === 'latest_reminder') {
      // OpenAI 兼容服务普遍允许中途插入 system；保留快照语义。
      result.push({ role: 'system', content: message.content })
      continue
    }
    if (message.role === 'user') {
      result.push({ role: 'user', content: await toUserContent(message.content, load) })
      continue
    }
    const text = textOf(message.content)
    if (text) {
      result.push({ role: 'assistant', content: text })
    }
  }
  return result
}

/** 只有 OpenAI 的 o 系列与 GPT-5 系列认识 reasoning_effort，其他兼容服务下发会被拒绝 */
const sendsReasoningEffort = ({ presetId }: StreamOptions['provider'], model: string) =>
  (presetId === 'openai' || presetId === 'openai-responses') && /^(o\d|gpt-5)/i.test(model)

const readUsage = (value: unknown): MessageUsage | undefined => {
  if (!isRecord(value)) {
    return undefined
  }
  const input = asNumber(value.prompt_tokens) ?? asNumber(value.input_tokens)
  const output = asNumber(value.completion_tokens) ?? asNumber(value.output_tokens)
  if (input === undefined && output === undefined) {
    return undefined
  }
  const cached =
    asNumber(at(value, ['prompt_tokens_details', 'cached_tokens'])) ??
    // DeepSeek 用命中的缓存 token 数
    asNumber(value.prompt_cache_hit_tokens) ??
    0
  return usageOf(input ?? 0, output ?? 0, cached)
}

async function* stream(options: StreamOptions): AsyncGenerator<MessageStreamEvent> {
  const { provider, deepseek, request: chat, loadAttachment, signal } = options
  const messages = await toMessages(chat.messages, loadAttachment)
  const body: Record<string, unknown> = {
    model: chat.model,
    messages: [{ role: 'system', content: chat.system }, ...messages],
    stream: true,
  }
  if (deepseek) {
    // DeepSeek 的 Chat Completions 与 Messages 一致：思考、服务端工具、安全校验和标题都在服务端完成
    body.thinking = { type: chat.thinking ? 'enabled' : 'disabled' }
    body.tools = [
      { type: 'deepseek_read_url', name: 'deepseek_read_url' },
      ...(chat.search ? [{ type: 'deepseek_web_search', name: 'deepseek_web_search' }] : []),
    ]
    body.safety_check = { level: 'high' }
    if (chat.title) {
      body.title = { enabled: true }
    }
  } else if (sendsReasoningEffort(provider, chat.model)) {
    // 思考等级直接映射到 OpenAI 的 reasoning_effort；关闭时给最低档，避免模型长时间思考。
    body.reasoning_effort = chat.thinking ? chat.thinkingLevel : 'minimal'
  }
  // Chat Completions 没有标准的服务端搜索工具，非 DeepSeek 保持不声明 tools，搜索由客户端工具服务补齐。
  const response = await request(`${baseUrl(provider.baseUrl)}/chat/completions`, {
    method: 'POST',
    headers: { ...authorization(provider.apiKey), Accept: 'text/event-stream' },
    body: JSON.stringify(body),
    signal,
  })
  const contentType = response.headers.get('Content-Type') ?? ''
  if (!response.body || !contentType.includes('text/event-stream')) {
    throw await readErrorResponse(response, 'protocol', '')
  }

  const blocks = new BlockStream()
  let usage: MessageUsage | undefined
  let stopReason: StopReason | null = null
  let done = false
  for await (const { event: eventName, data } of readServerSentEvents(response.body, signal)) {
    if (data.trim() === '[DONE]') {
      done = true
      break
    }
    let payload: unknown
    try {
      payload = JSON.parse(data)
    } catch {
      continue
    }
    if (isErrorBody(payload)) {
      throw new ApiError('stream', payload.error.message)
    }
    if (eventName === 'error') {
      throw new ApiError('stream', data)
    }
    if (!isRecord(payload)) {
      continue
    }
    usage = readUsage(payload.usage) ?? usage
    const choice = Array.isArray(payload.choices) ? payload.choices[0] : undefined
    if (!isRecord(choice)) {
      continue
    }
    const delta = isRecord(choice.delta) ? choice.delta : undefined
    if (delta) {
      // DeepSeek / Moonshot 用 reasoning_content，OpenRouter 用 reasoning。
      yield* blocks.thinking(
        asString(delta.reasoning_content) ?? asString(delta.reasoning) ?? undefined
      )
      yield* blocks.text(asString(delta.content))
    }
    stopReason = mapFinishReason(asString(choice.finish_reason)) ?? stopReason
    if (stopReason) {
      break
    }
  }
  if (!stopReason && !done) {
    throw new ApiError('interrupted', '')
  }
  yield* blocks.finish(stopReason ?? 'end_turn', usage)
}

const modelIds = (body: unknown): string[] => {
  const data = isRecord(body) ? body.data : undefined
  return Array.isArray(data)
    ? data.flatMap((item) => {
        const id = isRecord(item) ? asString(item.id) : undefined
        return id ? [id] : []
      })
    : []
}

const listModels = async (provider: { baseUrl: string; apiKey: string }, signal?: AbortSignal) => {
  const response = await request(`${baseUrl(provider.baseUrl)}/models`, {
    method: 'GET',
    headers: { Authorization: `Bearer ${provider.apiKey}` },
    signal,
  })
  return modelIds(await response.json())
}

export const openaiAdapter: ProtocolAdapter = {
  id: 'openai',
  stream,
  async verify(provider, signal) {
    await listModels(provider, signal)
  },
  listModels,
}
