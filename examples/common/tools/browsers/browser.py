# coding: utf-8
# Copyright (c) 2025 inclusionAI.

import base64
import json
import os
import subprocess
import time
import traceback
from importlib import resources
from pathlib import Path
from typing import Any, Dict, Tuple, List, Union

from aworld.config import ConfigDict
from examples.common.tools.common import package
from examples.common.tools.tool_action import BrowserAction
from aworld.core.common import Observation, ActionModel, ActionResult
from aworld.logs.util import logger
from aworld.core.tool.base import action_executor, ToolFactory
from aworld.core.tool.base import Tool
from aworld.utils.import_package import is_package_installed
from examples.common.tools.browsers.action.executor import BrowserToolActionExecutor
from examples.common.tools.browsers.util.dom import DomTree
from examples.common.tools.conf import BrowserToolConfig
from examples.common.tools.browsers.util.dom_build import build_dom_tree
from aworld.utils import import_package
from aworld.tools.utils import build_observation

URL_MAX_LENGTH = 4096
UTF8 = "".join(chr(x) for x in range(0, 55290))
ASCII = "".join(chr(x) for x in range(32, 128))

BROWSER = "browser"


@ToolFactory.register(name=BROWSER,
                      desc="browser",
                      supported_action=BrowserAction,
                      conf_file_name=f'browser_tool.yaml')
class BrowserTool(Tool):
    def __init__(self, conf: Union[ConfigDict, BrowserToolConfig], **kwargs) -> None:
        super(BrowserTool, self).__init__(conf, **kwargs)

        self.initialized = False
        self._finish = False
        self.record_trace = self.conf.get("enable_recording", False)
        self.sleep_after_init = self.conf.get("sleep_after_init", False)

        dom_js_path = self.conf.get('dom_js_path')
        if dom_js_path and os.path.exists(dom_js_path):
            with open(dom_js_path, 'r') as read:
                self.js_code = read.read()
        else:
            self.js_code = resources.read_text(f'{package}.browsers.script',
                                               'buildDomTree.js')
        self.cur_observation = None
        if not is_package_installed('playwright'):
            import_package("playwright")
            logger.info("playwright install...")
            try:
                subprocess.check_call('playwright install', shell=True, timeout=300)
            except Exception as e:
                logger.error(f"Fail to auto execute playwright install, you can install manually\n {e}")

    def init(self) -> None:
        from playwright.sync_api import sync_playwright

        if self.initialized:
            return

        self.context_manager = sync_playwright()
        self.playwright = self.context_manager.start()
        self.browser = self._create_browser()
        self.browser_context = self._create_browser_context()

        if self.record_trace:
            self.browser_context.tracing.start(screenshots=True, snapshots=True)

        self.page = self.browser_context.new_page()
        if self.conf.get("custom_executor"):
            self.action_executor = BrowserToolActionExecutor(self)
        else:
            self.action_executor = action_executor
        self.initialized = True

    def _create_browser(self):
        browse_name = self.conf.get("browse_name", "chromium")
        browse = getattr(self.playwright, browse_name)
        cdp_url = self.conf.get("cdp_url")
        wss_url = self.conf.get("wss_url")
        if cdp_url:
            if browse_name != "chromium":
                logger.warning(f"{browse_name} unsupported CDP, will use chromium browser")
                browse = self.playwright.chromium
            logger.info(f"Connecting to remote browser via CDP {cdp_url}")
            browser = browse.connect_over_cdp(cdp_url)
        elif wss_url:
            logger.info(f"Connecting to remote browser via wss {wss_url}")
            browser = browse.connect(wss_url)
        else:
            headless = self.conf.get("headless", False)
            slow_mo = self.conf.get("slow_mo", 0)
            disable_security_args = []
            if self.conf.get('disable_security', False):
                disable_security_args = ['--disable-web-security',
                                         '--disable-site-isolation-trials',
                                         '--disable-features=IsolateOrigins,site-per-process']
            args = ['--no-sandbox',
                    '--disable-crash-reporte',
                    '--disable-blink-features=AutomationControlled',
                    '--disable-infobars',
                    '--disable-background-timer-throttling',
                    '--disable-popup-blocking',
                    '--disable-backgrounding-occluded-windows',
                    '--disable-renderer-backgrounding',
                    '--disable-window-activation',
                    '--disable-focus-on-load',
                    '--no-first-run',
                    '--no-default-browser-check',
                    '--no-startup-window',
                    '--window-position=0,0',
                    '--window-size=1280,720'] + disable_security_args
            browser = browse.launch(
                headless=headless,
                slow_mo=slow_mo,
                args=args,
                proxy=self.conf.get('proxy'),
            )
        return browser

    def _create_browser_context(self):
        """Creates a new browser context with anti-detection measures and loads cookies if available."""
        from playwright.sync_api import ViewportSize

        browser = self.browser
        if self.conf.get("cdp_url") and len(browser.contexts) > 0:
            context = browser.contexts[0]
        else:
            viewport_size = ViewportSize(width=self.conf.get("width", 1280),
                                         height=self.conf.get("height", 720))
            disable_security = self.conf.get('disable_security', False)

            context = browser.new_context(viewport=viewport_size,
                                          no_viewport=False,
                                          user_agent=self.conf.get('user_agent'),
                                          java_script_enabled=True,
                                          bypass_csp=disable_security,
                                          ignore_https_errors=disable_security,
                                          record_video_dir=self.conf.get('working_dir'),
                                          record_video_size=viewport_size,
                                          locale=self.conf.get('locale'),
                                          storage_state=self.conf.get("storage_state", None),
                                          geolocation=self.conf.get("geolocation", None),
                                          device_scale_factor=1)
            if "chromium" == self.conf.get("browse_name", "chromium"):
                context.grant_permissions(['camera', 'microphone'])

        if self.conf.get('working_dir'):
            context.tracing.start(screenshots=True, snapshots=True, sources=True)

        cookie_file = self.conf.get('cookies_file')
        if cookie_file and os.path.exists(cookie_file):
            with open(cookie_file, 'r') as read:
                cookies = json.loads(read.read())
                context.add_cookies(cookies)
                logger.info(f'Cookies load from {cookie_file} finished')

        if self.conf.get('private'):
            js = resources.read_text(f"{package}.browsers.script", "stealth.min.js")
            context.add_init_script(js)

        return context

    def get_cur_page(self):
        return self.page

    def screenshot(self, full_page: bool = False) -> str:
        """Returns a base64 encoded screenshot of the current page.

        Args:
            full_page: When true, takes a screenshot of the full scrollable page, instead of the currently visible viewport.

        Returns:
            Base64 of the page screenshot
        """
        page = self.get_cur_page()

        try:
            page.bring_to_front()
            page.wait_for_load_state(timeout=2000)
        except:
            logger.warning("bring to front load timeout")
            pass

        screenshot = page.screenshot(
            full_page=full_page,
            animations='disabled',
            timeout=600000
        )
        logger.info("page screenshot finished")
        screenshot_base64 = base64.b64encode(screenshot).decode('utf-8')
        return screenshot_base64

    def _get_observation(self, info: Dict[str, Any] = {}) -> Observation:
        fail_error = info.get('exception')
        if fail_error:
            return Observation(observer=self.name(), action_result=[ActionResult(error=fail_error)])

        try:
            dom_tree = self._parse_dom_tree()
            image = self.screenshot()
            pixels_above, pixels_below = self._scroll_info()
            info.update({"pixels_above": pixels_above,
                         "pixels_below": pixels_below,
                         "url": self.page.url})
            return Observation(observer=self.name(),
                               dom_tree=dom_tree,
                               image=image,
                               info=info)
        except Exception as e:
            try:
                self.page.go_back()
            except:
                logger.warning("current page abnormal, new page to use.")
                self.page = self.browser_context.new_page()
            try:
                dom_tree = self._parse_dom_tree()
                image = self.screenshot()
                pixels_above, pixels_below = self._scroll_info()
                info.update({"pixels_above": pixels_above,
                             "pixels_below": pixels_below,
                             "url": self.page.url})
                return Observation(observer=self.name(), dom_tree=dom_tree, image=image, info=info)
            except Exception as e:
                logger.warning(f"build observation fail, {traceback.format_exc()}")
                return Observation(observer=self.name(), action_result=[ActionResult(error=traceback.format_exc())])

    def _parse_dom_tree(self) -> DomTree:
        args = {
            'doHighlightElements': self.conf.get("do_highlight", True),
            'focusHighlightIndex': self.conf.get("focus_highlight", -1),
            'viewportExpansion': self.conf.get("viewport_expansion", 0),
            'debugMode': logger.getEffectiveLevel() == 10,
        }
        element_tree, element_map = build_dom_tree(self.page, self.js_code, args)
        return DomTree(element_tree=element_tree, element_map=element_map)

    def _scroll_info(self) -> tuple[int, int]:
        """Get scroll position information for the current page."""
        scroll_y = self.page.evaluate('window.scrollY')
        viewport_height = self.page.evaluate('window.innerHeight')
        total_height = self.page.evaluate('document.documentElement.scrollHeight')
        pixels_above = scroll_y
        pixels_below = total_height - (scroll_y + viewport_height)
        return pixels_above, pixels_below

    def reset(self, *, seed: int | None = None, options: Dict[str, str] | None = None) -> Tuple[
        Observation, Dict[str, Any]]:
        super().reset(seed=seed, options=options)
        if self.initialized:
            observation = self._get_observation()
            observation.action_result = [ActionResult(content='start', keep=True)]
            self.cur_observation = observation
            return observation, {}

        self.close()
        self.init()

        if self.sleep_after_init > 0:
            time.sleep(self.sleep_after_init)

        observation = self._get_observation()
        observation.action_result = [ActionResult(content='start', keep=True)]
        self.cur_observation = observation
        return observation, {}

    @property
    def finished(self) -> bool:
        return self._finish

    def save_trace(self, trace_path: str | Path) -> None:
        if self.record_trace:
            self.browser_context.tracing.stop(path=trace_path)

    def close(self) -> None:
        if hasattr(self, 'context') and self.browser_context:
            self.browser_context.close()
        if hasattr(self, 'browser') and self.browser:
            self.browser.close()
        if hasattr(self, 'playwright') and self.playwright:
            self.playwright.stop()

        if self.initialized:
            self.context_manager.__exit__()

    def do_step(self, action: List[ActionModel], **kwargs) -> Tuple[
        Observation, float, bool, bool, Dict[str, Any]]:
        if not self.initialized:
            raise RuntimeError("Call init first before calling step.")

        if not action:
            logger.warning(f"{self.name()} has no action")
            return build_observation(observer=self.name(), ability='', content='no action'), 0., False, False, {}

        reward = 0
        fail_error = ""
        action_result = None

        invalid_acts: List[int] = []
        for i, act in enumerate(action):
            if act.tool_name != BROWSER:
                logger.warning(f"tool {act.tool_name} is not a browser!")
                invalid_acts.append(i)

        if invalid_acts:
            for i in invalid_acts:
                action[i] = None

        try:
            action_result, self.page = self.action_executor.execute_action(action,
                                                                           observation=self.cur_observation,
                                                                           llm_config=self.conf.llm_config,
                                                                           **kwargs)
            reward = 1
        except Exception as e:
            fail_error = str(e)

        info = {"exception": fail_error}
        terminated = kwargs.get("terminated", False)
        if action_result:
            for res in action_result:
                if res.is_done:
                    terminated = res.is_done
                    info['done'] = True
                    self._finish = True
                if res.error:
                    fail_error += res.error

        contains_write_to_file = any(act.action_name == BrowserAction.WRITE_TO_FILE.value.name for act in action if act)
        if contains_write_to_file:
            msg = ""
            for action_result_elem in action_result:
                msg = action_result_elem.content
            # write_to_file observation
            return (Observation(content=msg, action_result=action_result, info=info),
                    reward,
                    terminated,
                    kwargs.get("truncated", False),
                    info)
        elif fail_error:
            # failed error observation
            return (Observation(action_result=action_result, observer=self.name()),
                    reward,
                    terminated,
                    kwargs.get("truncated", False),
                    info)
        else:
            # normal observation
            observation = self._get_observation(info)
            observation.ability = action[-1].action_name
            observation.action_result = action_result
            self.cur_observation = observation
            return (observation,
                    reward,
                    terminated,
                    kwargs.get("truncated", False),
                    info)
