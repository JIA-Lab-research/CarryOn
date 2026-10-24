"""Built-in ``web_query`` skill for internet-backed lookup.

The skill intentionally has no API-key dependency. Weather queries use wttr.in's
JSON endpoint; general web lookups scrape DuckDuckGo's HTML results and return a
small structured summary.
"""
from __future__ import annotations
import html
import re
import time
from html.parser import HTMLParser
from typing import Any, Optional
from urllib.parse import parse_qs, quote, unquote, urlparse
import httpx
from ..types import ToolResult
_WEATHER_HINTS = ('天气', '气温', '温度', '下雨', '降雨', '空气质量', 'weather', 'temperature', 'rain')

class WebQuerySkill:
    name = 'web_query'

    def __init__(self, *, client: httpx.AsyncClient | None=None, timeout_s: float=10.0, max_results: int=5) -> None:
        self._client = client
        self._timeout_s = float(timeout_s)
        self._max_results = int(max_results)

    async def run(self, instruction: str, inputs: dict) -> ToolResult:
        started = time.time()
        raw = inputs or {}
        query = str(raw.get('query') or raw.get('q') or instruction or '').strip()
        if not query:
            return ToolResult(ok=False, error='web_query requires a non-empty instruction or inputs.query')
        kind = str(raw.get('kind') or raw.get('type') or '').strip().lower()
        if kind not in {'weather', 'search'}:
            kind = 'weather' if _looks_like_weather(query) else 'search'
        try:
            if self._client is None:
                async with httpx.AsyncClient(follow_redirects=True) as client:
                    data = await self._run_with_client(kind, query, raw, client)
            else:
                data = await self._run_with_client(kind, query, raw, self._client)
        except httpx.HTTPError as e:
            return ToolResult(ok=False, error=f'web query http error: {e}')
        except ValueError as e:
            return ToolResult(ok=False, error=f'web query parse error: {e}')
        return ToolResult(ok=True, data=data, latency_ms=int((time.time() - started) * 1000))

    async def _run_with_client(self, kind: str, query: str, inputs: dict, client: httpx.AsyncClient) -> dict:
        if kind == 'weather':
            return await self._weather(query, inputs, client)
        return await self._search(query, inputs, client)

    async def _weather(self, query: str, inputs: dict, client: httpx.AsyncClient) -> dict:
        explicit_location = str(inputs.get('location') or '').strip()
        location = explicit_location or _extract_weather_location(query)
        inferred = not bool(explicit_location)
        loc_part = quote(location, safe='') if location else ''
        url = f'https://wttr.in/{loc_part}?format=j1&lang=zh'
        resp = await client.get(url, headers=_headers(), timeout=self._timeout_s)
        resp.raise_for_status()
        payload = resp.json()
        current = _first(payload.get('current_condition'))
        today = _first(payload.get('weather'))
        nearest = _first(payload.get('nearest_area'))
        area_name = _area_name(nearest) or location or '当前位置'
        condition = _condition_text(current) or '未知'
        temp = current.get('temp_C') or '?'
        feels = current.get('FeelsLikeC') or '?'
        humidity = current.get('humidity') or '?'
        wind = current.get('windspeedKmph') or '?'
        uv = current.get('uvIndex')
        max_temp = today.get('maxtempC') or '?'
        min_temp = today.get('mintempC') or '?'
        rain = _chance_of_rain(today)
        summary = f'{area_name}：天气 {condition}，温度 {temp}°C，体感 {feels}°C，湿度 {humidity}%，风速 {wind} km/h。今天最高 {max_temp}°C，最低 {min_temp}°C'
        if rain is not None:
            summary += f'，降水概率 {rain}%'
        if uv is not None:
            summary += f'，UV 指数 {uv}'
        summary += '。'
        return {'kind': 'weather', 'query': query, 'location': area_name, 'requested_location': explicit_location or location, 'location_inferred': inferred, 'summary': summary, 'current': {'condition': condition, 'temp_c': _maybe_number(temp), 'feels_like_c': _maybe_number(feels), 'humidity_percent': _maybe_number(humidity), 'wind_kmph': _maybe_number(wind), 'uv_index': _maybe_number(uv)}, 'today': {'max_temp_c': _maybe_number(max_temp), 'min_temp_c': _maybe_number(min_temp), 'chance_of_rain_percent': _maybe_number(rain)}, 'source': url}

    async def _search(self, query: str, inputs: dict, client: httpx.AsyncClient) -> dict:
        max_results = _coerce_int(inputs.get('max_results'), self._max_results)
        max_results = max(1, min(max_results, 10))
        url = 'https://duckduckgo.com/html/'
        resp = await client.get(url, params={'q': query}, headers=_headers(), timeout=self._timeout_s)
        resp.raise_for_status()
        parser = _DuckDuckGoParser()
        parser.feed(resp.text)
        results = parser.results[:max_results]
        if not results:
            raise ValueError('no search results parsed from DuckDuckGo response')
        summary = '\n'.join((f"{idx}. {item['title']} - {item.get('snippet') or item['url']}" for idx, item in enumerate(results, 1)))
        return {'kind': 'search', 'query': query, 'summary': summary, 'results': results, 'source': str(resp.url)}

    async def aclose(self) -> None:
        return None

class _DuckDuckGoParser(HTMLParser):

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.results: list[dict[str, str]] = []
        self._mode: str | None = None
        self._buf: list[str] = []
        self._href = ''

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        attr = {k: v or '' for k, v in attrs}
        classes = set((attr.get('class') or '').split())
        if tag == 'a' and 'result__a' in classes:
            self._mode = 'title'
            self._buf = []
            self._href = _normalize_ddg_url(attr.get('href') or '')
        elif 'result__snippet' in classes:
            self._mode = 'snippet'
            self._buf = []

    def handle_data(self, data: str) -> None:
        if self._mode:
            self._buf.append(data)

    def handle_endtag(self, tag: str) -> None:
        if self._mode == 'title' and tag == 'a':
            title = _clean_text(''.join(self._buf))
            if title and self._href:
                self.results.append({'title': title, 'url': self._href, 'snippet': ''})
            self._reset()
        elif self._mode == 'snippet' and tag in {'a', 'div'}:
            snippet = _clean_text(''.join(self._buf))
            if snippet and self.results and (not self.results[-1].get('snippet')):
                self.results[-1]['snippet'] = snippet
            self._reset()

    def _reset(self) -> None:
        self._mode = None
        self._buf = []
        self._href = ''

def _looks_like_weather(query: str) -> bool:
    q = query.lower()
    return any((h in q for h in _WEATHER_HINTS))

def _extract_weather_location(query: str) -> str:
    text = str(query or '').strip()
    text = re.sub('(请|帮我|帮忙|查一下|查查|查询|联网|看一下|告诉我|一下)', '', text)
    text = re.sub('(今天|今日|明天|后天|现在|当前|当地|本地|天气预报|天气|气温|温度|会不会下雨|下雨|降雨|预报|怎么样|如何|怎样|吗|[？?])', '', text)
    text = re.sub('\\s+', '', text).strip('的在：:，,。.')
    if not text or len(text) > 40:
        return ''
    return text

def _first(value: Any) -> dict:
    if isinstance(value, list) and value and isinstance(value[0], dict):
        return value[0]
    return {}

def _area_name(area: dict) -> str:
    for key in ('areaName', 'region', 'country'):
        value = area.get(key)
        if isinstance(value, list) and value:
            item = value[0]
            if isinstance(item, dict) and item.get('value'):
                return str(item['value'])
    return ''

def _condition_text(current: dict) -> str:
    for key in ('lang_zh', 'weatherDesc'):
        value = current.get(key)
        if isinstance(value, list) and value:
            item = value[0]
            if isinstance(item, dict) and item.get('value'):
                return str(item['value'])
    return ''

def _chance_of_rain(today: dict) -> str | None:
    hourly = today.get('hourly')
    if not isinstance(hourly, list) or not hourly:
        return None
    values = []
    for item in hourly:
        if isinstance(item, dict) and item.get('chanceofrain') is not None:
            value = _coerce_int(item.get('chanceofrain'), -1)
            if value >= 0:
                values.append(value)
    if not values:
        return None
    return str(max(values))

def _maybe_number(value: Any) -> int | float | None:
    if value is None:
        return None
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    return int(num) if num.is_integer() else num

def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default

def _clean_text(value: str) -> str:
    return re.sub('\\s+', ' ', html.unescape(value or '')).strip()

def _normalize_ddg_url(href: str) -> str:
    if not href:
        return ''
    href = html.unescape(href)
    if href.startswith('//'):
        href = 'https:' + href
    elif href.startswith('/'):
        href = 'https://duckduckgo.com' + href
    parsed = urlparse(href)
    params = parse_qs(parsed.query)
    uddg = params.get('uddg')
    if uddg and uddg[0]:
        return unquote(uddg[0])
    return href

def _headers() -> dict[str, str]:
    return {'User-Agent': 'CarryOn/0.1 (+https://github.com/) httpx', 'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8'}
