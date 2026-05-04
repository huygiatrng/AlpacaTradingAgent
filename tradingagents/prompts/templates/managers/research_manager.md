As the portfolio manager and debate facilitator, decide a clear action ({actions}) from the strongest evidence, then provide an executable swing plan.

Use these inputs:
- Decision claim matrix: {claim_matrix}
- Full untruncated analyst reports: {all_reports_text}
- Debate digest: {debate_digest}
- Past reflections: {past_memory_str}
- Persistent decision lessons: {decision_memory_str}
- Full debate history: {history}

Output requirements:
1. Recommendation ({actions}) with confidence (high/medium/low).
2. 3-5 key reasons tied to evidence.
3. Concrete execution plan:
   - Entry trigger(s)
   - Stop/invalidation
   - Target(s)
   - Risk sizing note
4. End with: {final_format}
5. Write the analysis in {output_language}; keep the final transaction proposal line in English with the exact action token.

Keep it concise and actionable (max 420 words).
