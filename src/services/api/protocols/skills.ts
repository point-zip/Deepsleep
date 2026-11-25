import { isWebUrl } from '../../../utils/url.js'
import type {
  ServerToolUseBlock,
  TextCitation,
  WebSearchResult,
  WebSearchToolResultBlock,
} from '../types.js'
import { isRecord } from './json.js'

/**
 * 标准协议的联网搜索没有 DeepSeek 的扩展字段。
 * 这里把各协议原生的搜索调用与来源归一成现有的 server_tool_use / web_search_tool_result 块，
 * 界面继续按 kind 与 input.action 展示，不需要为新协议单独改 UI。
 */

/** 标准搜索工具名；DeepSeek 扩展使用 deepseek_web_search */
export const WEB_SEARCH_NAME = 'web_search'

/** 一条可展示的来源：只保留 HTTP(S) 链接，避免把归档或本地路径渲染成来源卡片 */
export type SkillSource = { url: string; title: string }

const firstString = (value: Record<string, unknown>, keys: string[]) => {
  for (const key of keys) {
    const item = value[key]
    if (typeof item === 'string' && item.trim()) {
      return item.trim()
    }
  }
  return undefined
}

const sourceOf = (value: Record<string, unknown>): SkillSource | null => {
  const url = firstString(value, ['url', 'uri', 'link'])
  if (!url || !isWebUrl(url)) {
    return null
  }
  return { url, title: firstString(value, ['title', 'name', 'display_name']) ?? url }
}

/**
 * 从 Responses 的 url_citation / web_search_call.action.sources 以及 Gemini 的
 * groundingChunks 等结构里宽松地取出来源；不认识的结构返回空数组，不影响后续解析。
 */
export const readSkillSources = (value: unknown): SkillSource[] => {
  if (Array.isArray(value)) {
    return value.flatMap(readSkillSources)
  }
  if (!isRecord(value)) {
    return []
  }
  const direct = sourceOf(value)
  const nested = [
    value.url_citation,
    value.annotation,
    value.annotations,
    value.web,
    value.source,
    value.citation,
    value.groundingChunks,
  ].flatMap(readSkillSources)
  return direct ? [direct, ...nested] : nested
}

/** 同一 URL 只保留第一次出现的来源，标题以后续更完整的为准由调用方决定 */
export const uniqueSkillSources = (sources: SkillSource[]): SkillSource[] => {
  const byUrl = new Map<string, SkillSource>()
  for (const source of sources) {
    if (!byUrl.has(source.url)) {
      byUrl.set(source.url, source)
    }
  }
  return [...byUrl.values()]
}

/** 把来源并入列表：URL 去重；已有标题只是 URL 时，用后续注解里的真实标题补齐 */
export const mergeSkillSources = (target: SkillSource[], incoming: SkillSource[]) => {
  for (const source of incoming) {
    const existing = target.find((item) => item.url === source.url)
    if (!existing) {
      target.push(source)
    } else if (
      (!existing.title || existing.title === existing.url) &&
      source.title &&
      source.title !== source.url
    ) {
      existing.title = source.title
    }
  }
}

/** 兼容 query / queries / q 几种写法，供 Anthropic 的 input_json 与 Responses 的 action 使用 */
export const readSearchQueries = (value: unknown): string[] => {
  const queries: string[] = []
  const add = (item: unknown) => {
    if (typeof item === 'string' && item.trim()) {
      queries.push(item.trim())
    }
  }
  if (typeof value === 'string') {
    add(value)
  } else if (Array.isArray(value)) {
    value.forEach(add)
  } else if (isRecord(value)) {
    add(value.query)
    add(value.q)
    if (Array.isArray(value.queries)) {
      value.queries.forEach(add)
    }
  }
  return [...new Set(queries)]
}

/** 标准搜索调用：input 在 content_block_start 时就已完整，界面不依赖 input_json_delta */
export const searchToolUse = (id: string, queries: string[]): ServerToolUseBlock => {
  const unique = [...new Set(queries.map((query) => query.trim()).filter(Boolean))]
  return {
    type: 'server_tool_use',
    id,
    name: 'web_search',
    input: {
      action: 'search',
      ...(unique[0] ? { query: unique[0] } : {}),
      queries: unique,
    },
    caller: { type: 'direct' },
    kind: 'search',
  }
}

/**
 * 标准搜索结果块。
 * failed 仅在没有可用来源时使用：成功但零结果按空数组处理，界面显示「未找到」而不是失败。
 */
export const searchToolResult = (
  id: string,
  sources: SkillSource[],
  failed = false
): WebSearchToolResultBlock => {
  const content: WebSearchToolResultBlock['content'] = failed
    ? [{ type: 'web_search_tool_result_error', error_code: 'unavailable' }]
    : uniqueSkillSources(sources).map<WebSearchResult>(({ url, title }) => ({
        type: 'web_search_result',
        url,
        title: title || url,
        snippet: '',
        site_icon: null,
        site_name: null,
        encrypted_content: '',
        page_age: null,
      }))
  return { type: 'web_search_tool_result', tool_use_id: id, content, kind: 'search' }
}

/** 搜索块挂不上时（如 Gemini 只在最后一个分片给 grounding），退化成文本引用，至少保住来源 */
export const sourceCitation = ({ url, title }: SkillSource): TextCitation => ({
  type: 'web_search_result_location',
  url,
  title,
  cited_text: '',
  encrypted_index: '',
})
