"""
Prompt Capture Utility for TradingAgents WebUI

This module provides utilities to capture and store the prompts used by each agent
when generating reports, allowing users to view the exact prompts via "Show Prompt" buttons.
"""

import re
from typing import Dict, Optional, Any
from webui.utils.state import app_state
from tradingagents.run_logger import get_run_audit_logger


class PromptCapture:
    """Utility class for capturing and managing agent prompts"""
    
    @staticmethod
    def extract_system_message_from_prompt(prompt_template) -> str:
        """
        Extract the system message from a LangChain ChatPromptTemplate or similar structure
        """
        try:
            # Handle LangChain ChatPromptTemplate
            if hasattr(prompt_template, 'messages'):
                for message in prompt_template.messages:
                    if hasattr(message, 'prompt') and hasattr(message.prompt, 'template'):
                        template = message.prompt.template
                        # Look for system message content
                        if 'system_message' in template or 'You are' in template:
                            return template
                    elif hasattr(message, 'content') and isinstance(message.content, str):
                        if 'You are' in message.content or 'system_message' in message.content:
                            return message.content
                            
            # Handle direct string prompts
            elif isinstance(prompt_template, str):
                return prompt_template
                
            # Handle dict-like structures
            elif hasattr(prompt_template, '__dict__'):
                for key, value in prompt_template.__dict__.items():
                    if isinstance(value, str) and ('You are' in value or len(value) > 100):
                        return value
                        
        except Exception as e:
            print(f"[PROMPT_CAPTURE] Error extracting prompt: {e}")
            
        return "Prompt extraction failed - format not recognized"
    
    @staticmethod
    def extract_prompt_from_agent_file(agent_file_path: str, agent_type: str) -> str:
        """
        Extract the system message directly from an agent file by reading its content
        """
        try:
            with open(agent_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Look for system_message variable assignment
            system_msg_pattern = r'system_message\s*=\s*\(\s*["\']([^"\']*(?:["\'][^"\']*)*)["\']'
            match = re.search(system_msg_pattern, content, re.DOTALL)
            
            if match:
                system_message = match.group(1)
                # Clean up the extracted message
                system_message = re.sub(r'\s+', ' ', system_message.strip())
                return system_message
                
            # Fallback: look for any long string that might be a prompt
            long_string_pattern = r'["\']([^"\']{200,})["\']'
            matches = re.findall(long_string_pattern, content, re.DOTALL)
            
            if matches:
                # Return the longest string found (likely the main prompt)
                longest_match = max(matches, key=len)
                return re.sub(r'\s+', ' ', longest_match.strip())
                
        except Exception as e:
            print(f"[PROMPT_CAPTURE] Error reading agent file {agent_file_path}: {e}")
            
        return f"Could not extract prompt from {agent_type} agent file"
    
    @staticmethod
    def get_default_prompts() -> Dict[str, str]:
        """
        Return fallback prompts when no real prompt has been captured yet
        """
        return {
            "market_report": "Market analyst prompt not yet captured from live execution",
            "sentiment_report": "Social media analyst prompt not yet captured from live execution", 
            "news_report": "News analyst prompt not yet captured from live execution",
            "fundamentals_report": "Fundamentals analyst prompt not yet captured from live execution",
            "macro_report": "Macro analyst prompt not yet captured from live execution",
            "bull_report": "Bull researcher prompt not yet captured from live execution",
            "bear_report": "Bear researcher prompt not yet captured from live execution", 
            "research_manager_report": "Research manager prompt not yet captured from live execution",
            "trader_investment_plan": "Trader prompt not yet captured from live execution"
        }
    
    @staticmethod
    def store_prompt_for_report(report_type: str, prompt_content: str, symbol: str = None):
        """
        Store a prompt for a specific report type in the application state
        """
        try:
            app_state.store_agent_prompt(report_type, prompt_content, symbol)
            get_run_audit_logger().log_prompt(
                report_type=report_type,
                prompt_text=prompt_content,
                symbol=symbol,
                metadata={"source": "prompt_capture"},
            )
        except Exception as e:
            print(f"[PROMPT_CAPTURE] Error storing prompt for {report_type}: {e}")
    
    @staticmethod
    def get_prompt_for_report(report_type: str, symbol: str = None) -> Optional[str]:
        """
        Get the stored prompt for a specific report type
        """
        try:
            stored_prompt = app_state.get_agent_prompt(report_type, symbol)
            if stored_prompt:
                return stored_prompt
                
            # Fallback to default prompts if no stored prompt found
            default_prompts = PromptCapture.get_default_prompts()
            return default_prompts.get(report_type, f"No prompt available for {report_type}")
            
        except Exception as e:
            print(f"[PROMPT_CAPTURE] Error getting prompt for {report_type}: {e}")
            return f"Error retrieving prompt for {report_type}"


# Convenience function for easy import
def capture_agent_prompt(report_type: str, prompt_content: str, symbol: str = None):
    """Convenience function to capture an agent's prompt"""
    PromptCapture.store_prompt_for_report(report_type, prompt_content, symbol)


def get_agent_prompt(report_type: str, symbol: str = None) -> str:
    """Convenience function to get an agent's prompt"""
    return PromptCapture.get_prompt_for_report(report_type, symbol) or "No prompt available" 
