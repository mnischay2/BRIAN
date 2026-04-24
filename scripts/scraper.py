import asyncio
from playwright.async_api import async_playwright

try:
    from logger import setup_aether_logger
except:
    from .logger import setup_aether_logger

logger = setup_aether_logger("Scraper")

class WebScraper:
    async def scrape(self, url: str) -> str:
        logger.info(f"Scraping URL: {url}")
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                context = await browser.new_context(
                    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                )
                page = await context.new_page()
                
                # Avoid dialogs and popups
                page.on("dialog", lambda d: d.dismiss())
                await page.add_init_script("window.alert = () => {}; window.confirm = () => true;")

                await page.goto(url, wait_until="domcontentloaded", timeout=30000)
                
                # Wait for network to settle
                try:
                    await page.wait_for_load_state("networkidle", timeout=5000)
                except:
                    pass

                # Extract and clean text
                text = await page.evaluate("document.body.innerText")
                await browser.close()
                
                cleaned = " ".join(text.split())
                return cleaned[:10000] # Delimit size for LLM context by using -- return cleaned
        except Exception as e:
            logger.error(f"Failed to scrape {url}: {e}")
            return ""
        

def test_scraper():
    scraper = WebScraper()
    url = "https://bodh.sarthhakai.com/"
    content = asyncio.run(scraper.scrape(url))
    print(content)  # Print first 500 chars of scraped content

if __name__ == "__main__":
    test_scraper()