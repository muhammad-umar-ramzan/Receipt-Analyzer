# AI Advice section - Simple version
st.divider()
st.subheader("🤖 AI Financial Insights")

groq_api_key = os.getenv("GROQ_API_KEY")

if groq_api_key:
    try:
        client = Groq(api_key=groq_api_key)
        
        analysis_text = "\n".join([
            f"- {cat}: €{amt:.2f}" 
            for cat, amt in category_totals.items()
        ])
        
        prompt = f"""
        Analyze this receipt:
        
        Total: €{total_spend:.2f}
        Categories:
        {analysis_text}
        
        Items:
        {df[['Item', 'Price']].to_string(index=False)}
        
        Provide brief financial advice (max 100 words):
        """
        
        with st.spinner("Generating insights..."):
            response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a helpful financial advisor."},
                    {"role": "user", "content": prompt}
                ],
                model="llama-3.1-8b-instant",
                temperature=0.7,
                max_tokens=200
            )
            
            advice = response.choices[0].message.content
            
            # Simple solution: use st.info or st.success
            st.success("💡 Financial Advice:")
            st.markdown(advice)  # Simple markdown without custom styling
            
            # OR use an expander
            with st.expander("View Financial Advice", expanded=True):
                st.write(advice)
    
    except Exception as e:
        st.info("AI insights temporarily unavailable")
else:
    st.info("💡 Set up GROQ_API_KEY for AI-powered insights")
