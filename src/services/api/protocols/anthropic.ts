import {
  ApiError,
  isErrorBody,
  originOf,
  readErrorResponse,
  request,
  trimBaseUrl,
} from '../client.js'
import { readMessageEvent } from '../parseEvent.js'
import { readServerSentEvents } from '../sse.js'
import type {
  AssistantBlockParam,
  FileBlockParam,
  FileObject,
  FilePurpose,
  MessageParam,
  MessageStreamEvent,
  ThinkingLevel,
  WebSearchResult,
  WebSearchToolResultBlock,
} from '../types.js'
import { asString, isRecord } from './json.js'
import { readSearchQueries, searchToolUse, WEB_SEARCH_NAME } from './skills.js'
import type { AttachmentLoader, ProtocolAdapter, StreamOptions } from './types.js'
import { UNAVAILABLE_ATTACHMENT_TEXT } from './types.js'

/** Anthropic Messages 接口的版本头，官方服务必传；DeepSeek 兼容接口不需要 */
const ANTHROPIC_VERSION = '2023-06-01'
/** 引用 Files API 上传的图片、调用 Files API 时必须携带 */
const FILES_BETA = 'files-api-2025-04-14'
/** 官方 Anthropic 必须给 max_tokens，且要大于思考预算 */
const MAX_TOKENS = 8192
/** 思考等级到预算的映射；官方 Messages 只接受 budget_tokens */
const THINKING_BUDGET: Record<ThinkingLevel, number> = {
  minimal: 1024,
  low: 2048,
  medium: 4096,
  high: 8192,
  xhigh: 16384,
  max: 32768,
}

const baseUrl = (base: string) => trimBaseUrl(base)

const messageHeaders = (
  apiKey: string,
  options: { deepseek: boolean; beta: boolean }
): Record<string, string> => ({
  // DeepSeek 兼容接口的鉴权方式，也可使用 Authorization: Bearer。
  'x-api-key': apiKey,
  ...(options.deepseek ? {} : { 'anthropic-version': ANTHROPIC_VERSION }),
  // Files API 和通过 file_id 引用图片时必传；纯文本聊天无需携带。
  ...(options.beta ? { 'anthropic-beta': FILES_BETA } : {}),
})

/** 文件块：file 源原样下发；local 源读取本地原件后内联为 base64，读不到就跳过 */
const fileBlock = async (block: FileBlockParam, load: AttachmentLoader) => {
  if (block.source.type === 'file') {
    // 只回传协议字段，本地原件信息（local）不能出现在请求里
    return { type: block.type, source: block.source }
  }
  const loaded = await load({ id: block.source.id, mediaType: block.source.media_type })
  return loaded
    ? {
        type: block.type,
        source: { type: 'base64', media_type: loaded.mediaType, data: loaded.base64 },
      }
    : null
}

const toUserContent = async (
  content: string | (FileBlockParam | { type: 'text'; text: string })[],
  load: AttachmentLoader
) => {
  if (typeof content === 'string') {
    return content
  }
  const blocks = []
  for (const block of content) {
    if (block.type === 'text') {
      if (block.text) {
        blocks.push(block)
      }
      continue
    }
    const converted = await fileBlock(block, load)
    if (converted) {
      blocks.push(converted)
    }
  }
  return blocks.length > 0 ? blocks : UNAVAILABLE_ATTACHMENT_TEXT
}

/**
 * assistant 内容块：DeepSeek 扩展按原样回传思考与工具块；
 * 其他 Anthropic 兼容服务只接受文本，思考签名与工具块对它们无效。
 */
const toAssistantContent = (
  content: Extract<MessageParam, { role: 'assistant' }>['content'],
  deepseek: boolean
) => {
  const blocks: AssistantBlockParam[] = []
  for (const block of content) {
    if (block.type === 'text') {
      if (block.text) {
        blocks.push(block)
      }
    } else if (deepseek) {
      blocks.push(block)
    }
  }
  return blocks
}

const toMessages = async (messages: MessageParam[], deepseek: boolean, load: AttachmentLoader) => {
  const result: Record<string, unknown>[] = []
  for (const message of messages) {
    if (message.role === 'latest_reminder') {
      // 新版官方接口不接受该角色，只有 DeepSeek 扩展认识。
      if (deepseek) {
        result.push(message)
      }
      continue
    }
    if (message.role === 'user') {
      result.push({ role: 'user', content: await toUserContent(message.content, load) })
      continue
    }
    const content = toAssistantContent(message.content, deepseek)
    if (content.length > 0) {
      result.push({ role: 'assistant', content })
    }
  }
  return result
}

/** 阅读链接始终开启，智能搜索开关控制是否另行声明搜索工具（含边想边搜）。 */
const tools = (search: boolean) => [
  { type: 'deepseek_read_url', name: 'deepseek_read_url' },
  ...(search ? [{ type: 'deepseek_web_search', name: 'deepseek_web_search' }] : []),
]

const indexOf = (value: unknown) =>
  typeof value === 'number' && Number.isSafeInteger(value) && value >= 0 ? value : undefined

const parsePartialJson = (value: string): unknown => {
  if (!value.trim()) {
    return undefined
  }
  try {
    return JSON.parse(value)
  } catch {
    return undefined
  }
}

/** 官方 web_search_tool_result 的 content 既可能是结果数组，也可能是单个错误对象 */
const readStandardSearchResult = (value: unknown): WebSearchToolResultBlock | null => {
  if (!isRecord(value)) {
    return null
  }
  const toolUseId = asString(value.tool_use_id)
  if (!toolUseId) {
    return null
  }
  const items =
    value.content === undefined
      ? []
      : Array.isArray(value.content)
        ? value.content
        : [value.content]
  const content: WebSearchToolResultBlock['content'] = []
  for (const item of items) {
    if (!isRecord(item)) {
      continue
    }
    if (item.type === 'web_search_tool_result_error') {
      // 官方错误码比本地多（max_uses_exceeded 等），界面只区分成功与失败，统一归一。
      content.push({ type: 'web_search_tool_result_error', error_code: 'unavailable' })
      continue
    }
    const url = asString(item.url)
    if (item.type !== 'web_search_result' || !url) {
      continue
    }
    const result: WebSearchResult = {
      type: 'web_search_result',
      url,
      title: asString(item.title) ?? url,
      snippet: asString(item.snippet) ?? '',
      page_age: asString(item.page_age) ?? null,
      encrypted_content: asString(item.encrypted_content) ?? '',
      site_icon: null,
      site_name: null,
    }
    content.push(result)
  }
  return { type: 'web_search_tool_result', tool_use_id: toolUseId, content, kind: 'search' }
}

type PendingSearchUse = { id: string; inputJson: string }

/**
 * 官方 Anthropic 的搜索事件与 DeepSeek 扩展不完全一致：
 * server_tool_use 在 content_block_start 里没有 input，查询参数随后经 input_json_delta 下发；
 * 这里先缓存，等块结束时再合成一个携带完整 input 的规范块。
 */
class StandardAnthropicSkills {
  private readonly pending = new Map<number, PendingSearchUse>()

  private nextId = 0

  private id() {
    this.nextId += 1
    return `srvtoolu_web_${this.nextId}`
  }

  read(payload: unknown): MessageStreamEvent[] {
    if (!isRecord(payload)) {
      return []
    }
    const index = indexOf(payload.index)
    if (payload.type === 'content_block_start' && index !== undefined) {
      const block = payload.content_block
      if (isRecord(block) && block.type === 'server_tool_use') {
        if (asString(block.name) === WEB_SEARCH_NAME) {
          const id = asString(block.id) ?? this.id()
          const queries = readSearchQueries(block.input)
          if (queries.length > 0) {
            return [
              { type: 'content_block_start', index, content_block: searchToolUse(id, queries) },
            ]
          }
          this.pending.set(index, { id, inputJson: '' })
        }
        // 其他标准服务端工具（如 web_fetch）暂不展示，交给后续事件自然结束。
        return []
      }
      if (isRecord(block) && block.type === 'web_search_tool_result') {
        const result = readStandardSearchResult(block)
        return result ? [{ type: 'content_block_start', index, content_block: result }] : []
      }
    }
    if (payload.type === 'content_block_delta' && index !== undefined) {
      const pending = this.pending.get(index)
      if (pending && isRecord(payload.delta) && payload.delta.type === 'input_json_delta') {
        pending.inputJson += asString(payload.delta.partial_json) ?? ''
        return []
      }
    }
    if (payload.type === 'content_block_stop' && index !== undefined) {
      const pending = this.pending.get(index)
      if (pending) {
        this.pending.delete(index)
        return [
          {
            type: 'content_block_start',
            index,
            content_block: searchToolUse(
              pending.id,
              readSearchQueries(parsePartialJson(pending.inputJson))
            ),
          },
          { type: 'content_block_stop', index },
        ]
      }
    }
    if (
      payload.type === 'message_delta' &&
      isRecord(payload.delta) &&
      payload.delta.stop_reason === 'pause_turn'
    ) {
      // 官方长搜索会返回 pause_turn；当前界面没有续跑状态，按正常结束收尾，避免丢弃整个 message_delta。
      const event = readMessageEvent({
        ...payload,
        delta: { ...payload.delta, stop_reason: 'end_turn' },
      })
      return event ? [event] : []
    }
    const event = readMessageEvent(payload)
    return event ? [event] : []
  }
}

const requestBody = (
  { request, deepseek }: Pick<StreamOptions, 'request' | 'deepseek'>,
  messages: Record<string, unknown>[]
): Record<string, unknown> => {
  const body: Record<string, unknown> = {
    model: request.model,
    system: request.system,
    messages,
    stream: true,
  }
  if (deepseek) {
    body.thinking = { type: request.thinking ? 'enabled' : 'disabled' }
    body.tools = tools(request.search)
    body.safety_check = { level: 'high' }
    if (request.title) {
      body.title = { enabled: true }
    }
    return body
  }
  const budget = THINKING_BUDGET[request.thinkingLevel]
  body.max_tokens = Math.max(MAX_TOKENS, budget + 4096)
  if (request.thinking) {
    body.thinking = { type: 'enabled', budget_tokens: budget }
  }
  if (request.search) {
    // 标准 Anthropic 服务端搜索工具；查询、抓取都在服务端完成，客户端只解析结果。
    body.tools = [{ type: 'web_search_20250305', name: 'web_search' }]
  }
  return body
}

/** 解析 Anthropic 兼容的 SSE：事件名与 JSON 的 type 相同，以 message_stop 结束 */
async function* stream(options: StreamOptions): AsyncGenerator<MessageStreamEvent> {
  const { provider, deepseek, request: chat, loadAttachment, signal } = options
  const messages = await toMessages(chat.messages, deepseek, loadAttachment)
  const body = requestBody(options, messages)
  const hasFile = chat.messages.some(
    (message) =>
      message.role !== 'latest_reminder' &&
      typeof message.content !== 'string' &&
      message.content.some((block) => block.type === 'image' || block.type === 'document')
  )
  const response = await request(`${baseUrl(provider.baseUrl)}/messages`, {
    method: 'POST',
    headers: {
      ...messageHeaders(provider.apiKey, { deepseek, beta: hasFile }),
      'Content-Type': 'application/json',
      Accept: 'text/event-stream',
    },
    body: JSON.stringify(body),
    signal,
  })
  const contentType = response.headers.get('Content-Type') ?? ''
  if (!response.body || !contentType.includes('text/event-stream')) {
    throw await readErrorResponse(response, 'protocol', '')
  }
  // DeepSeek 扩展继续走原有解析；标准协议额外归一内置搜索事件。
  const skills = deepseek ? null : new StandardAnthropicSkills()
  for await (const { event: eventName, data } of readServerSentEvents(response.body, signal)) {
    let payload: unknown
    try {
      payload = JSON.parse(data)
    } catch {
      if (eventName !== 'error') {
        continue
      }
    }
    if (isErrorBody(payload)) {
      throw new ApiError('stream', payload.error.message)
    }
    if (eventName === 'error') {
      throw new ApiError('stream', data)
    }
    const events = skills ? skills.read(payload) : [readMessageEvent(payload)]
    for (const event of events) {
      if (!event) {
        continue
      }
      yield event
      if (event.type === 'message_stop') {
        return
      }
    }
  }
  // 服务端没有提供错误内容时保持为空，不推测断流原因。
  throw new ApiError('interrupted', '')
}

/** GET /models 的候选地址：官方 Anthropic 在 /v1/models，DeepSeek 在服务根下 */
const modelEndpoints = (provider: {
  baseUrl: string
  apiKey: string
}): { url: string; headers: Record<string, string> }[] => {
  const base = baseUrl(provider.baseUrl)
  return [
    {
      url: `${base}/models`,
      headers: { 'x-api-key': provider.apiKey, 'anthropic-version': ANTHROPIC_VERSION },
    },
    {
      url: `${originOf(base)}/models`,
      headers: { Authorization: `Bearer ${provider.apiKey}` },
    },
  ]
}

const isMissing = (error: unknown) =>
  error instanceof ApiError && (error.status === 404 || error.status === 405)

const requestModels = async (
  provider: { baseUrl: string; apiKey: string },
  signal: AbortSignal | undefined,
  parse: (body: unknown) => string[]
): Promise<string[]> => {
  let failure: unknown
  for (const endpoint of modelEndpoints(provider)) {
    try {
      const response = await request(endpoint.url, {
        method: 'GET',
        headers: endpoint.headers,
        signal,
      })
      return parse(await response.json())
    } catch (error) {
      if (!isMissing(error)) {
        throw error
      }
      failure = error
    }
  }
  throw failure instanceof ApiError
    ? failure
    : new ApiError('protocol', 'Model list is unavailable')
}

const readModelIds = (body: unknown): string[] => {
  if (typeof body !== 'object' || body === null || !('data' in body)) {
    return []
  }
  const data = (body as { data: unknown }).data
  return Array.isArray(data)
    ? data.flatMap((item) => {
        const id =
          typeof item === 'object' && item !== null && 'id' in item
            ? (item as { id: unknown }).id
            : undefined
        return typeof id === 'string' && id ? [id] : []
      })
    : []
}

/**
 * POST {base}/files 上传附件（不设过期时间，会话里的历史附件一直可用）。
 * purpose 为 null 时不传该字段（图片）；purpose 必须排在 file 之前。
 */
export const uploadFile = async (
  provider: { baseUrl: string; apiKey: string; deepseek: boolean },
  name: string,
  body: Blob,
  purpose: FilePurpose | null,
  signal?: AbortSignal
): Promise<FileObject> => {
  const form = new FormData()
  if (purpose) {
    form.append('purpose', purpose)
  }
  form.append('file', body, name)
  const response = await request(`${baseUrl(provider.baseUrl)}/files`, {
    method: 'POST',
    headers: messageHeaders(provider.apiKey, {
      deepseek: provider.deepseek,
      beta: true,
    }),
    body: form,
    signal,
  })
  return (await response.json()) as FileObject
}

export const anthropicAdapter: ProtocolAdapter = {
  id: 'anthropic',
  stream,
  async verify(provider, signal) {
    await requestModels(provider, signal, () => [])
  },
  listModels(provider, signal) {
    return requestModels(provider, signal, readModelIds)
  },
}
