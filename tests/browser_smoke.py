"""Explicit browser check: python -m pytest tests/browser_smoke.py -v.

Uses installed Chrome and the real FastAPI routes with the suite's offline LLM
fixtures. CDN assets still require network access. Screenshots go to pytest's tmpdir.
"""

import pytest
from fastapi.testclient import TestClient
from playwright.sync_api import sync_playwright, expect
from app.main import app


@pytest.mark.parametrize("width,height", [(1440, 1080), (390, 844)])
def test_risk_desk_and_signoff_journey(width, height, tmp_path):
    with TestClient(app) as client, sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="chrome", headless=True)
        page = browser.new_page(viewport={"width": width, "height": height}, reduced_motion="reduce")
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))

        def serve(route):
            request = route.request
            response = client.request(request.method, request.url.replace("http://apex.test", ""),
                                      headers={"Content-Type": request.headers.get("content-type", "")}, content=request.post_data)
            route.fulfill(status=response.status_code, content_type=response.headers.get("content-type"), body=response.content)

        page.route("http://apex.test/**", serve)
        page.goto("http://apex.test/", wait_until="networkidle")
        expect(page.locator("#headerActiveTicker")).to_have_text("NVDA")
        risk_tab = page.get_by_role("button", name="F5 RISK DESK").filter(visible=True)
        risk_tab.focus()
        page.keyboard.press("Enter")
        expect(page.locator("#scenarioRows tr")).to_have_count(3)
        expect(page.locator("#scenarioSource")).to_contain_text("Illustrative sample")
        region = page.get_by_role("region", name="Assumption stress comparison")
        region.focus()
        expect(region).to_be_focused()
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth"), "Table must not widen the page"
        if width < 500:
            assert region.evaluate("el => el.scrollWidth > el.clientWidth"), "Mobile table should scroll locally"
        page.evaluate("scrollTo(0, 0)")
        page.screenshot(path=str(tmp_path / f"risk-desk-{width}.png"), full_page=True)

        page.get_by_role("button", name="Run committee").click()
        expect(page.locator("#runStatus")).to_contain_text("sign-off", timeout=15000)
        page.get_by_role("button", name="Review order").click()
        size = page.get_by_label("Size override")
        assert float(size.get_attribute("max")) == float(size.input_value())
        # Bypass the HTML maximum to exercise the server boundary from the UI.
        size.fill("15")
        page.get_by_role("button", name="Approve", exact=True).click()
        expect(page.locator("#gateError")).to_contain_text("only reduce")
        page.get_by_label("Size override").fill("1")
        page.get_by_label("Decision notes").fill("Browser-tested smaller position")
        page.get_by_role("button", name="Approve", exact=True).click()
        expect(page.locator("#pmGateCard")).to_contain_text("Target Allocation: 1")
        expect(page.locator("#hitlStatus")).to_contain_text("approved")

        page.get_by_role("button", name="Run committee").click()
        expect(page.locator("#runStatus")).to_contain_text("sign-off", timeout=15000)
        page.get_by_role("button", name="Review order").click()
        page.get_by_label("Decision notes").fill("<script>reject</script>")
        page.get_by_role("button", name="Reject", exact=True).click()
        expect(page.locator("#pmGateCard")).to_contain_text("ORDER CANCELLED")
        expect(page.locator("#pmGateCard")).to_contain_text("<script>reject</script>")
        assert page.locator("#pmGateCard script").count() == 0
        assert errors == []
        browser.close()
