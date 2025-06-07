"""Compression scheduling utilities"""

from typing import Dict, Any

class CompressionScheduler:
    """Schedule compression parameters over time"""
    
    def __init__(self, schedule_config: Dict[str, Any]):
        self.schedule_config = schedule_config
        self.current_step = 0
    
    def step(self):
        """Advance the scheduler by one step"""
        self.current_step += 1
    
    def get_current_config(self) -> Dict[str, Any]:
        """Get current compression configuration"""
        return self.schedule_config
