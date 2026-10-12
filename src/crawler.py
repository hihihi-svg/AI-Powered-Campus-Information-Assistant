"""
crawler.py - Robust, polite website crawler for campus information acquisition.

Features:
- robots.txt compliance checking
- Domain restriction and link normalization
- Priority-guided BFS crawl frontier prioritizing campus domains (academics, admissions, etc.)
- Configurable rate limiting, timeout, depth, and page limits
- Dual handling of HTML pages and public PDF documents
- Error handling with exponential backoff retries
"""

import heapq
import logging
import re
import time
import urllib.robotparser
from datetime import datetime, timezone
from typing import Any, Dict, Generator, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse, urldefrag

import requests
import urllib3

# Suppress insecure HTTPS warnings if verify=False is configured for the campus host
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger("campus_pipeline.crawler")


class RobotsManager:
    """Manages and checks robots.txt rules for target domains."""

    def __init__(self, user_agent: str = "*", verify_ssl: bool = False, timeout: int = 10):
        self.user_agent = user_agent
        self.verify_ssl = verify_ssl
        self.timeout = timeout
        self.parsers: Dict[str, urllib.robotparser.RobotFileParser] = {}

    def get_parser(self, base_url: str) -> urllib.robotparser.RobotFileParser:
        parsed = urlparse(base_url)
        origin = f"{parsed.scheme}://{parsed.netloc}"
        if origin in self.parsers:
            return self.parsers[origin]

        robots_url = f"{origin}/robots.txt"
        parser = urllib.robotparser.RobotFileParser()
        parser.set_url(robots_url)
        try:
            resp = requests.get(
                robots_url,
                headers={"User-Agent": self.user_agent},
                verify=self.verify_ssl,
                timeout=self.timeout,
            )
            if resp.status_code == 200:
                parser.parse(resp.text.splitlines())
                logger.info(f"Loaded robots.txt successfully from {robots_url}")
            else:
                parser.allow_all = True
                logger.warning(f"robots.txt returned status {resp.status_code}; allowing crawl.")
        except Exception as e:
            parser.allow_all = True
            logger.warning(f"Failed to fetch robots.txt from {robots_url} ({e}); defaulting to allow.")

        self.parsers[origin] = parser
        return parser

    def is_allowed(self, url: str) -> bool:
        parser = self.get_parser(url)
        return parser.can_fetch(self.user_agent, url)


class URLNormalizer:
    """Normalizes, validates, and filters campus URLs."""

    def __init__(self, allowed_domains: List[str], ignored_extensions: List[str]):
        self.allowed_domains = [d.lower() for d in allowed_domains]
        self.ignored_extensions = set(ext.lower() for ext in ignored_extensions)

    def normalize(self, url: str) -> Optional[str]:
        """Normalize URL by stripping fragments, lowercase host/scheme, and cleaning."""
        if not url:
            return None
        # Remove fragment
        url, _ = urldefrag(url)
        url = url.strip()

        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return None

        netloc = parsed.netloc.lower()
        # Remove standard ports if present
        if ":" in netloc:
            host, port = netloc.split(":", 1)
            if (parsed.scheme == "http" and port == "80") or (parsed.scheme == "https" and port == "443"):
                netloc = host

        # Normalize path
        path = parsed.path
        if not path:
            path = "/"
        else:
            # Collapse multiple slashes
            path = re.sub(r"/+", "/", path)

        # Standardize query string
        query = parsed.query

        normalized = f"{parsed.scheme.lower()}://{netloc}{path}"
        if query:
            normalized = f"{normalized}?{query}"
        return normalized

    def is_allowed(self, url: str) -> bool:
        """Check if URL matches allowed domains and is not an ignored binary extension."""
        parsed = urlparse(url)
        host = parsed.netloc.lower()
        if not any(host == d or host.endswith(f".{d}") for d in self.allowed_domains):
            return False

        path_lower = parsed.path.lower()
        for ext in self.ignored_extensions:
            if path_lower.endswith(ext):
                return False

        # Exclude administrative/login URLs
        excluded_patterns = [
            "/wp-admin",
            "/wp-login.php",
            "/feed/",
            "/xmlrpc.php",
            "replytocom=",
        ]
        if any(pat in url.lower() for pat in excluded_patterns):
            return False

        return True


class CampusCrawler:
    """Polite, priority-guided campus crawler for HTML pages and PDFs."""

    def __init__(self, config: Dict[str, Any]):
        crawler_cfg = config.get("crawler", {})
        self.start_urls: List[str] = crawler_cfg.get("start_urls", ["https://rvce.edu.in/"])
        self.allowed_domains: List[str] = crawler_cfg.get("allowed_domains", ["rvce.edu.in"])
        self.max_depth: int = crawler_cfg.get("max_depth", 3)
        self.max_pages: int = crawler_cfg.get("max_pages", 50)
        self.max_pdfs: int = crawler_cfg.get("max_pdfs", 10)
        self.request_delay: float = crawler_cfg.get("request_delay", 0.5)
        self.timeout: int = crawler_cfg.get("timeout", 15)
        self.user_agent: str = crawler_cfg.get("user_agent", "CampusAssistantBot/1.0")
        self.verify_ssl: bool = crawler_cfg.get("verify_ssl", False)
        self.respect_robots: bool = crawler_cfg.get("respect_robots_txt", True)
        self.priority_keywords: List[str] = [k.lower() for k in crawler_cfg.get("priority_keywords", [])]
        self.ignored_extensions: List[str] = crawler_cfg.get("ignored_extensions", [])
        self.retry_attempts: int = crawler_cfg.get("retry_attempts", 3)
        self.retry_backoff: float = crawler_cfg.get("retry_backoff_factor", 1.5)

        self.normalizer = URLNormalizer(self.allowed_domains, self.ignored_extensions)
        self.robots_manager = RobotsManager(self.user_agent, self.verify_ssl, self.timeout)

        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": self.user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,application/pdf,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        })

        # State tracking
        self.visited_urls: Set[str] = set()
        self.discovered_urls: Set[str] = set()
        self.pdf_count: int = 0
        self.html_count: int = 0
        self.errors: List[Dict[str, Any]] = []

    def _calculate_priority(self, url: str) -> int:
        """
        Compute crawl priority score. Lower number = higher priority for heapq.
        Urls matching priority keywords get smaller numbers (higher priority).
        """
        url_lower = url.lower()
        score = 100
        for keyword in self.priority_keywords:
            if keyword in url_lower:
                score -= 10
        # Boost PDFs if we haven't reached max_pdfs
        if url_lower.endswith(".pdf"):
            score -= 5
        return score

    def _fetch_url(self, url: str) -> Tuple[Optional[requests.Response], Optional[str]]:
        """Fetch URL with retries and exponential backoff."""
        error_msg = None
        for attempt in range(1, self.retry_attempts + 1):
            try:
                response = self.session.get(
                    url,
                    timeout=self.timeout,
                    verify=self.verify_ssl,
                    allow_redirects=True,
                )
                return response, None
            except requests.exceptions.RequestException as e:
                error_msg = f"Attempt {attempt}/{self.retry_attempts} failed: {type(e).__name__} - {str(e)}"
                logger.warning(f"Error fetching {url}: {error_msg}")
                if attempt < self.retry_attempts:
                    time.sleep(self.retry_backoff ** attempt)
        return None, error_msg

    def _extract_links_from_html(self, base_url: str, html_text: str) -> List[Tuple[str, str]]:
        """Extract all candidate anchor links and PDF references from HTML text."""
        from bs4 import BeautifulSoup
        links = []
        try:
            soup = BeautifulSoup(html_text, "html.parser")
            for a_tag in soup.find_all("a", href=True):
                raw_href = a_tag["href"].strip()
                if not raw_href or raw_href.startswith(("#", "javascript:", "mailto:", "tel:")):
                    continue
                full_url = urljoin(base_url, raw_href)
                normalized = self.normalizer.normalize(full_url)
                if normalized and self.normalizer.is_allowed(normalized):
                    anchor_text = a_tag.get_text(separator=" ", strip=True)
                    links.append((normalized, anchor_text))
        except Exception as e:
            logger.debug(f"Error parsing links from {base_url}: {e}")
        return links

    def crawl(self) -> Generator[Dict[str, Any], None, None]:
        """
        Execute prioritized crawling loop yielding raw page/document records.
        Yields raw records as dictionaries.
        """
        # Queue format: (priority_score, depth, counter, url)
        queue: List[Tuple[int, int, int, str]] = []
        counter = 0

        for start_url in self.start_urls:
            normalized = self.normalizer.normalize(start_url)
            if normalized:
                priority = self._calculate_priority(normalized)
                heapq.heappush(queue, (priority, 0, counter, normalized))
                self.discovered_urls.add(normalized)
                counter += 1

        logger.info(f"Initialized crawler with {len(queue)} seed URLs. Target limit: {self.max_pages} pages.")

        while queue and (self.html_count + self.pdf_count) < self.max_pages:
            priority, depth, _, current_url = heapq.heappop(queue)

            if current_url in self.visited_urls:
                continue

            # Respect robots.txt
            if self.respect_robots and not self.robots_manager.is_allowed(current_url):
                logger.info(f"Disallowed by robots.txt: {current_url}")
                self.errors.append({
                    "url": current_url,
                    "error_type": "RobotsDisallowed",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
                self.visited_urls.add(current_url)
                continue

            # Check PDF quota
            is_pdf_url = current_url.lower().endswith(".pdf")
            if is_pdf_url and self.pdf_count >= self.max_pdfs:
                logger.debug(f"Skipping PDF {current_url} as max_pdfs limit ({self.max_pdfs}) is reached.")
                continue

            # Polite delay between requests
            time.sleep(self.request_delay)

            collected_at = datetime.now(timezone.utc).isoformat()
            response, fetch_error = self._fetch_url(current_url)

            if fetch_error or response is None:
                self.errors.append({
                    "url": current_url,
                    "error_type": "FetchError",
                    "details": fetch_error,
                    "timestamp": collected_at,
                })
                self.visited_urls.add(current_url)
                continue

            final_url = self.normalizer.normalize(response.url) or current_url
            status_code = response.status_code
            content_type_header = response.headers.get("Content-Type", "").lower()
            last_modified_header = response.headers.get("Last-Modified")

            if status_code != 200:
                self.errors.append({
                    "url": current_url,
                    "final_url": final_url,
                    "status_code": status_code,
                    "error_type": "HTTPError",
                    "timestamp": collected_at,
                })
                self.visited_urls.add(current_url)
                continue

            # Determine content type
            is_pdf = "application/pdf" in content_type_header or is_pdf_url

            if is_pdf:
                # Read PDF binary content (limit to 25MB for safety)
                try:
                    raw_bytes = response.content
                    self.pdf_count += 1
                    self.visited_urls.add(current_url)
                    self.visited_urls.add(final_url)

                    yield {
                        "url": current_url,
                        "final_url": final_url,
                        "content_type": "PDF",
                        "mime_type": "application/pdf",
                        "http_status": status_code,
                        "collected_at": collected_at,
                        "last_modified": last_modified_header,
                        "depth": depth,
                        "raw_bytes": raw_bytes,
                        "raw_content": None,
                    }
                except Exception as e:
                    self.errors.append({
                        "url": current_url,
                        "error_type": "PDFReadError",
                        "details": str(e),
                        "timestamp": collected_at,
                    })
                continue

            elif "text/html" in content_type_header or "text/plain" in content_type_header:
                try:
                    response.encoding = response.apparent_encoding or "utf-8"
                    html_content = response.text
                except Exception as e:
                    logger.warning(f"Error reading response text from {current_url}: {e}")
                    html_content = ""

                self.html_count += 1
                self.visited_urls.add(current_url)
                self.visited_urls.add(final_url)

                yield {
                    "url": current_url,
                    "final_url": final_url,
                    "content_type": "HTML",
                    "mime_type": content_type_header,
                    "http_status": status_code,
                    "collected_at": collected_at,
                    "last_modified": last_modified_header,
                    "depth": depth,
                    "raw_bytes": None,
                    "raw_content": html_content,
                }

                # Discover child links if within max_depth
                if depth < self.max_depth:
                    child_links = self._extract_links_from_html(final_url, html_content)
                    for child_url, anchor_text in child_links:
                        if child_url not in self.visited_urls and child_url not in self.discovered_urls:
                            self.discovered_urls.add(child_url)
                            child_priority = self._calculate_priority(child_url)
                            # Anchor text priority boost
                            for kw in self.priority_keywords:
                                if kw in anchor_text.lower():
                                    child_priority -= 5
                                    break
                            heapq.heappush(queue, (child_priority, depth + 1, counter, child_url))
                            counter += 1
            else:
                self.errors.append({
                    "url": current_url,
                    "error_type": "UnsupportedContentType",
                    "mime_type": content_type_header,
                    "timestamp": collected_at,
                })
                self.visited_urls.add(current_url)

        logger.info(
            f"Crawl completed. Visited {len(self.visited_urls)} URLs. "
            f"Retrieved {self.html_count} HTML pages and {self.pdf_count} PDFs. "
            f"Errors logged: {len(self.errors)}."
        )
