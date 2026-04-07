"""WebFetchTool with SSRF protection."""
import ipaddress
import socket
from html.parser import HTMLParser
from urllib.parse import urlparse

import httpx


MAX_RESPONSE_SIZE = 1_048_576  # 1MB
FETCH_TIMEOUT = 30
BLOCKED_IP_RANGES = [
    ipaddress.ip_network('10.0.0.0/8'),
    ipaddress.ip_network('172.16.0.0/12'),
    ipaddress.ip_network('192.168.0.0/16'),
    ipaddress.ip_network('127.0.0.0/8'),
    ipaddress.ip_network('169.254.0.0/16'),  # Link-local / cloud metadata
    ipaddress.ip_network('::1/128'),
    ipaddress.ip_network('fc00::/7'),
    ipaddress.ip_network('fe80::/10'),
]


class _TextExtractor(HTMLParser):
    """Simple HTML to text converter."""
    def __init__(self):
        super().__init__()
        self.parts: list[str] = []
        self._skip = False
        self._skip_tags = {'script', 'style', 'head'}

    def handle_starttag(self, tag, attrs):
        if tag in self._skip_tags:
            self._skip = True

    def handle_endtag(self, tag):
        if tag in self._skip_tags:
            self._skip = False

    def handle_data(self, data):
        if not self._skip:
            text = data.strip()
            if text:
                self.parts.append(text)

    def get_text(self) -> str:
        return '\n'.join(self.parts)


def _is_private_ip(hostname: str) -> bool:
    """Check if hostname resolves to a private/blocked IP."""
    try:
        addr_info = socket.getaddrinfo(hostname, None)
        for _, _, _, _, sockaddr in addr_info:
            ip = ipaddress.ip_address(sockaddr[0])
            if any(ip in network for network in BLOCKED_IP_RANGES):
                return True
    except socket.gaierror:
        pass
    return False


class WebFetchTool:
    def __init__(self):
        pass

    def fetch_url(self, url: str, raw: bool = False) -> str:
        """Fetch a URL and return its content. Returns cleaned text by default, raw HTML if raw=True."""
        parsed = urlparse(url)
        if parsed.scheme not in ('http', 'https'):
            raise PermissionError(f'Only http/https URLs allowed, got: {parsed.scheme}')
        if not parsed.hostname:
            raise ValueError('Invalid URL: no hostname')
        if _is_private_ip(parsed.hostname):
            raise PermissionError(f'Access to private/internal IPs is blocked: {parsed.hostname}')

        with httpx.Client(timeout=FETCH_TIMEOUT, follow_redirects=True) as client:
            response = client.get(url)
            # Check size from headers first
            content_length = response.headers.get('content-length')
            if content_length and int(content_length) > MAX_RESPONSE_SIZE:
                raise ValueError(f'Response too large: {content_length} bytes (max {MAX_RESPONSE_SIZE})')
            # Read and check actual size
            content = response.text
            if len(content.encode()) > MAX_RESPONSE_SIZE:
                raise ValueError(f'Response too large (max {MAX_RESPONSE_SIZE} bytes)')

        if raw:
            return content

        extractor = _TextExtractor()
        extractor.feed(content)
        return extractor.get_text()

    def get_tools(self) -> list[tuple]:
        """Return tool specs as (name, description, parameters, callable) tuples."""
        return [(
            'fetch_url',
            'Fetch a URL and return its text content. Set raw=True for raw HTML.',
            {'url': str, 'raw': bool},
            self.fetch_url
        )]
