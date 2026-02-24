"""
State fix utility to correct report field mapping in sequential execution mode
"""

def apply_report_mapping_fix():
    """Apply fix to ensure correct report field mapping"""
    import os
    import sys
    
    # Add project root to path
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, project_root)
    
    try:
        from webui.utils.state import AppState
        
        # Patch the process_chunk_updates method to fix report mapping
        original_process_chunk_updates = AppState.process_chunk_updates
        
        def fixed_process_chunk_updates(self, chunk, ticker=None):
            """Fixed version that correctly maps social analyst reports"""
            
            # Fix report field mapping before processing
            if "sentiment_report" in chunk and chunk["sentiment_report"]:
                # Ensure we're not accidentally overwriting market_report
                # This happens in sequential mode when the graph incorrectly streams data
                # print(f"[FIX] Processing sentiment_report correctly")
                pass
                
            # Check for incorrect mapping (social analyst updating market_report)
            elif "market_report" in chunk and chunk["market_report"]:
                # Check if this is coming from Social Analyst (sequential bug)
                current_symbol = getattr(self, 'current_symbol', '')
                if current_symbol:
                    state = self.get_state(current_symbol)
                    if state and state["agent_statuses"].get("Social Analyst") == "in_progress":
                        # This is the bug! Social Analyst is incorrectly updating market_report
                        # print(f"[FIX] Detected Social Analyst incorrectly updating market_report - fixing...")
                        # Move the content to sentiment_report
                        chunk["sentiment_report"] = chunk["market_report"]
                        del chunk["market_report"]
                        # print(f"[FIX] Corrected: market_report -> sentiment_report")
            
            # Call the original method with the fixed chunk
            return original_process_chunk_updates(self, chunk, ticker=ticker)
        
        # Apply the patch
        AppState.process_chunk_updates = fixed_process_chunk_updates
        print("✅ Applied report mapping fix for sequential execution mode")
        return True
        
    except Exception as e:
        print(f"❌ Error applying fix: {e}")
        return False


if __name__ == "__main__":
    apply_report_mapping_fix() 