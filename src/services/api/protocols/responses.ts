import { ApiError, isErrorBody, readErrorResponse, request, trimBaseUrl } from '../client.js'
import { readServerSentEvents } from '../sse.js'
import type {
  FileBlockParam,
  MessageParam,
  MessageStreamEvent,
  MessageUsage,
  ServerToolBlock,
  ServerToolUseBlock,
  StopReason,
} from '../types.js'
import { BlockStream, usageOf } from './blocks.js'
import { dataUrl, localRef } from './inline.js'
import { asNumber, asString, at, isRecord } from './json.js'
import type { SkillSource } from './skills.js'
import {
  mergeSkillSources,
  readSearchQueries,
  readSkillSources,
  searchToolResult,
  searchToolUse,
  sourceCitation,
} from './skills.js'
import type { AttachmentLoader, ProtocolAdapter, StreamOptions } from './types.js'
import { UNAVAILABLE_ATTACHMENT_TEXT } from './types.js'

const baseUrl = (base: string) => trimBaseUrl(base)

const authorization = (apiKey: string) => ({
  Authorization: `Bearer ${apiKey}`,
  'Content-Type': 'application/json',
})

const textOf = (content: Extract<MessageParam, { role: 'assistant' }>['content']) =>
  content
    .flatMap((block) => (block.type === 'text' ? [block.text] : []))
    .join('')
    .trim()

const inputParts = async (
  blocks: (FileBlockParam | { type: 'text'; text: string })[],
  load: AttachmentLoader
) => {
  const parts: Record<string, unknown>[] = []
  for (const block of blocks) {
    if (block.type === 'text') {
      if (block.text) {
        parts.push({ type: 'input_text', text: block.text })
      }
      continue
    }
    const ref = localRef(block)
    const loaded = ref ? await load(ref) : null
    if (loaded && loaded.mediaType.startsWith('image/')) {
      parts.push({ type: 'input_image', image_url: dataUrl(loaded.mediaType, loaded.base64) })
    }
  }
  return parts
}

const toInput = async (messages: MessageParam[], load: AttachmentLoader) => {
  const input: Record<string, unknown>[] = []
  for (const message of messages) {
    if (message.role === 'latest_reminder') {
      input.push({ role: 'system', content: message.content })
      continue
    }
    if (message.role === 'user') {
      if (typeof message.content === 'string') {
        input.push({ role: 'user', content: message.content })
        continue
      }
      const parts = await inputParts(message.content, load)
      input.push({
        role: 'user',
        content:
          parts.length > 0 ? parts : [{ type: 'input_text', text: UNAVAILABLE_ATTACHMENT_TEXT }],
      })
      continue
    }
    const text = textOf(message.content)
    if (text) {
      input.push({ role: 'assistant', content: text })
    }
  }
  return input
}

const readUsage = (value: unknown): MessageUsage | undefined => {
  if (!isRecord(value)) {
    return undefined
  }
  const input = asNumber(value.input_tokens)
  const output = asNumber(value.output_tokens)
  if (input === undefined && output === undefined) {
    return undefined
  }
  return usageOf(
    input ?? 0,
    output ?? 0,
    asNumber(at(value, ['input_tokens_details', 'cached_tokens'])) ?? 0
  )
}

type ResponsesSearch = {
  id: string
  queries: string[]
  failed: boolean
  /** 调用块已经下发后不再重复插入，结果块在收尾时统一补齐 */
  emitted: boolean
}

/**
 * Responses 的内置搜索没有独立的结果列表：来源在 output_text 的 url_citation 注解里，
 * 或（请求 include 之后）在 web_search_call.action.sources 里。
 * 这里把 web_search_call 转成调用块，把来源聚合成结果块，供现有搜索面板展示。
 */
class ResponsesSkills {
  private readonly searches: ResponsesSearch[] = []

  private readonly sources: SkillSource[] = []

  private textStarted = false

  private nextId = 0

  /** 文本开始输出后不再插入调用块，否则会把已输出的正文拆成前置说明 */
  markText() {
    this.textStarted = true
  }

  private id() {
    this.nextId += 1
    return `ws_web_${this.nextId}`
  }

  private addSources(value: unknown) {
    mergeSkillSources(this.sources, readSkillSources(value))
  }

  private addQueries(search: ResponsesSearch, value: unknown) {
    for (const query of readSearchQueries(value)) {
      if (!search.queries.includes(query)) {
        search.queries.push(query)
      }
    }
  }

  private searchOf(id: string): ResponsesSearch {
    const existing = this.searches.find((item) => item.id === id)
    if (existing) {
      return existing
    }
    const created: ResponsesSearch = { id, queries: [], failed: false, emitted: false }
    this.searches.push(created)
    return created
  }

  /**
   * 处理 output_item 里的 web_search_call / message。
   * 返回需要立刻展示的调用块；in_progress 阶段没有 action 时先等待 output_item.done。
   * outputIndex 用于缺失 id 时保持 added / done 两个事件指向同一次调用。
   */
  readItem(value: unknown, final: boolean, outputIndex?: number): ServerToolUseBlock | null {
    if (!isRecord(value)) {
      return null
    }
    if (value.type === 'message') {
      this.addSources(value.content)
      this.addSources(value.annotations)
      return null
    }
    if (value.type !== 'web_search_call') {
      return null
    }
    const id =
      asString(value.id) ?? (outputIndex === undefined ? this.id() : `ws_web_index_${outputIndex}`)
    const search = this.searchOf(id)
    this.addQueries(search, value)
    if (isRecord(value.action)) {
      this.addQueries(search, value.action)
      this.addSources(value.action.sources)
      if (value.action.type === 'open_page') {
        this.addSources([{ url: value.action.url, title: value.action.title }])
      }
    }
    this.addSources(value.sources)
    search.failed ||= value.status === 'failed'
    if (search.emitted || this.textStarted || (!final && search.queries.length === 0)) {
      return null
    }
    search.emitted = true
    return searchToolUse(search.id, search.queries)
  }

  /** response.completed 里可能带完整 output，用来补齐流式事件漏掉的调用和引用 */
  readResponse(value: unknown): ServerToolUseBlock[] {
    if (!isRecord(value) || !Array.isArray(value.output)) {
      return []
    }
    const uses: ServerToolUseBlock[] = []
    for (const [index, item] of value.output.entries()) {
      const use = this.readItem(item, true, index)
      if (use) {
        uses.push(use)
      }
    }
    return uses
  }

  readCitations(value: unknown) {
    this.addSources(value)
  }

  /** 每个已展示的调用都要有结果块，否则界面会把缺失结果的搜索显示成失败 */
  resultBlocks(): ServerToolBlock[] {
    return this.searches
      .filter((search) => search.emitted)
      .map((search) =>
        searchToolResult(search.id, this.sources, search.failed && this.sources.length === 0)
      )
  }

  /** 没有任何调用块可挂结果时，把来源退化成文本引用，至少保住来源卡片 */
  fallbackSources(): SkillSource[] {
    return this.searches.some((search) => search.emitted) ? [] : [...this.sources]
  }
}

/** 流式事件名即响应类型；消费文本、思考、内置搜索与结束事件，其余忽略 */
async function* stream(options: StreamOptions): AsyncGenerator<MessageStreamEvent> {
  const { provider, deepseek, request: chat, loadAttachment, signal } = options
  const input = await toInput(chat.messages, loadAttachment)
  const body: Record<string, unknown> = {
    model: chat.model,
    instructions: chat.system,
    input,
    stream: true,
  }
  if (deepseek) {
    // DeepSeek 的 Responses 接口同样在服务端完成搜索、阅读链接与安全校验
    // Responses 里 low 仍然是思考，只有 none 才是关闭
    body.reasoning = { effort: chat.thinking ? 'high' : 'none' }
    body.tools = [
      { type: 'deepseek_read_url' },
      ...(chat.search ? [{ type: 'deepseek_web_search' }] : []),
    ]
    body.safety_check = { level: 'high' }
    if (chat.title) {
      body.title = { enabled: true }
    }
  } else if (
    (provider.presetId === 'openai' || provider.presetId === 'openai-responses') &&
    /^(o\d|gpt-5)/i.test(chat.model)
  ) {
    body.reasoning = { effort: chat.thinking ? chat.thinkingLevel : 'minimal' }
  }
  if (!deepseek && chat.search) {
    // 标准内置搜索工具；DeepSeek 分支在上面已声明自己的扩展工具。
    body.tools = [{ type: 'web_search' }]
    // 让服务端在 web_search_call.action 里带回来源，annotations 缺失时也能生成结果块。
    body.include = ['web_search_call.action.sources']
  }
  const response = await request(`${baseUrl(provider.baseUrl)}/responses`, {
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
  // DeepSeek 的 Responses 接口已有自己的工具块格式，不参与标准内置搜索的解析。
  const skills = deepseek ? null : new ResponsesSkills()
  for await (const { event: eventName, data } of readServerSentEvents(response.body, signal)) {
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
      throw new ApiError('stream', asString(at(payload, ['error', 'message'])) ?? data)
    }
    if (!isRecord(payload)) {
      continue
    }
    switch (asString(payload.type) ?? eventName) {
      case 'response.output_text.delta': {
        const delta = asString(payload.delta)
        if (delta) {
          skills?.markText()
        }
        skills?.readCitations(payload.annotations)
        yield* blocks.text(delta)
        break
      }
      case 'response.output_text.annotation.added':
        skills?.readCitations(payload.annotation)
        break
      case 'response.output_text.done':
      case 'response.content_part.done':
        skills?.readCitations(payload.annotations)
        if (isRecord(payload.part)) {
          skills?.readCitations(payload.part.annotations)
        }
        break
      case 'response.output_item.added':
      case 'response.output_item.done': {
        // in_progress 的 web_search_call 没有 action，等 output_item.done 再下发调用块。
        const use = skills?.readItem(
          payload.item,
          payload.type === 'response.output_item.done',
          asNumber(payload.output_index)
        )
        if (use) {
          yield* blocks.tool(use)
        }
        break
      }
      case 'response.reasoning_summary_text.delta':
      case 'response.reasoning_text.delta':
        yield* blocks.thinking(asString(payload.delta))
        break
      case 'response.completed':
      case 'response.incomplete': {
        const result = isRecord(payload.response) ? payload.response : undefined
        const usage = readUsage(result?.usage)
        let stopReason: StopReason = 'end_turn'
        if (asString(result?.status) === 'incomplete') {
          const reason = asString(at(result, ['incomplete_details', 'reason'])) ?? ''
          stopReason = reason.includes('max_output') ? 'max_tokens' : 'end_turn'
        }
        for (const use of skills?.readResponse(result) ?? []) {
          yield* blocks.tool(use)
        }
        // 来源通常比正文晚到；结果块可以追加在正文之后，view 会按 tool_use_id 覆盖回对应的搜索步骤。
        for (const block of skills?.resultBlocks() ?? []) {
          yield* blocks.tool(block)
        }
        for (const source of skills?.fallbackSources() ?? []) {
          yield* blocks.citation(sourceCitation(source))
        }
        yield* blocks.finish(stopReason, usage)
        return
      }
      case 'response.failed': {
        const message = asString(at(payload, ['response', 'error', 'message'])) ?? ''
        throw new ApiError('stream', message)
      }
      default:
        break
    }
  }
  throw new ApiError('interrupted', '')
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

export const responsesAdapter: ProtocolAdapter = {
  id: 'responses',
  stream,
  async verify(provider, signal) {
    await listModels(provider, signal)
  },
  listModels,
}
