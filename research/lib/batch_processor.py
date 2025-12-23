"""
Generic Claude Batch API Processor.
"""

import time
import logging
from typing import Dict, List, Any
import anthropic

class ClaudeBatchProcessor:
    """Generic Claude Batch API Processor."""

    def __init__(self, api_key: str, poll_interval: int = 30):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.poll_interval = poll_interval
        self.logger = logging.getLogger(self.__class__.__name__)

    def submit_batch(self, requests: List[Dict]) -> str:
        """Submit batch to Claude API."""
        self.logger.info(f"🚀 Submitting batch with {len(requests)} requests...")
        try:
            message_batch = self.client.beta.messages.batches.create(requests=requests)
            self.logger.info(f"✅ Batch submitted: {message_batch.id}")
            return message_batch.id
        except Exception as e:
            raise RuntimeError(f"Failed to submit batch: {e}")

    def wait_for_batch_completion(self, batch_id: str) -> Any:
        """Poll batch status until completion."""
        self.logger.info(f"⏳ Waiting for batch {batch_id}...")
        start_time = time.time()
        last_status = None

        while True:
            try:
                batch = self.client.beta.messages.batches.retrieve(batch_id)
                status = batch.processing_status

                if status != last_status:
                    self.logger.info(f"📊 Status: {status} (elapsed: {time.time() - start_time:.1f}s)")
                    if hasattr(batch, 'request_counts'):
                        counts = batch.request_counts
                        total = counts.total or 0
                        done = (counts.succeeded or 0) + (counts.errored or 0) + (counts.canceled or 0)
                        if total > 0:
                            self.logger.info(f"   Progress: {done}/{total} ({done/total*100:.1f}%)")
                    last_status = status

                if status in ["ended", "failed", "canceled", "expired"]:
                    return batch

                time.sleep(self.poll_interval)

            except Exception as e:
                self.logger.error(f"Error polling batch: {e}")
                time.sleep(self.poll_interval)

    def download_batch_results(self, batch_id: str) -> List[Dict]:
        """Download batch results."""
        self.logger.info("⬇️ Downloading results...")
        try:
            results = []
            for item in self.client.beta.messages.batches.results(batch_id):
                res_dict = {"custom_id": item.custom_id, "result": {}}
                if item.result:
                    res_dict["result"]["type"] = item.result.type
                    if item.result.type == "succeeded":
                        content = item.result.message.content
                        text = content[0].text if content else ""
                        res_dict["result"]["message"] = {"content": [{"text": text}]}
                    elif item.result.type == "errored":
                        res_dict["result"]["error"] = {"type": item.result.error.type}
                results.append(res_dict)
            
            self.logger.info(f"📥 Downloaded {len(results)} items")
            return results
        except Exception as e:
            raise RuntimeError(f"Failed to download results: {e}")
