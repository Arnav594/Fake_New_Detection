# Run this to create a sample
import pandas as pd

df = pd.read_excel("bharatfakenewskosh.xlsx")

# # Save first 500 rows as CSV
# df.head(350).to_csv("sample_data.csv", index=False)
# print("✅ CSV saved! Now download and upload it here")

# Show me 10 actual text samples
print("="*80)
print("SAMPLE DATA - PLEASE COPY THIS OUTPUT")
print("="*80)

for i in range(10):
    row = df.iloc[i]
    print(f"\n--- ROW {i+1} ---")
    print(f"Statement: {row['Eng_Trans_Statement']}")
    print(f"Body (first 200 chars): {str(row['Eng_Trans_News_Body'])[:200]}")
    print(f"Label: {row['Label']}")
    print("-"*80)
# ```

# ---

# ## 📊 **OR - Can You Answer These Questions?**

# Looking at your screenshot, I can see some text but it's cut off. Can you tell me:

# 1. **What does a typical "Statement" contain?**
#    - Is it the actual fake claim? (e.g., "Modi said XYZ")
#    - Or is it a description? (e.g., "A claim about Modi saying XYZ is viral")

# 2. **What does the "Body" contain?**
#    - The original fake news article?
#    - Or the fact-check explanation?

# 3. **What do TRUE vs FALSE actually mean?**
#    - TRUE = "This claim is factually correct"
#    - FALSE = "This claim is debunked/false"

# 4. **Where did this dataset come from?**
#    - Looks like "Pratyek" fact-checking organization?
#    - From their website/database?

# ---

# ## 🎯 **Based On What I See:**

# ### **My Hypothesis:**
# ```
# Statement: "A video viral on social media claims..."
# Body: "We fact-checked this. Here's what we found... 
#        The claim is FALSE because..."
# Label: FALSE

# vs.

# Statement: "Reports say government announced..."
# Body: "We verified this. Here's the evidence...
#        The claim is TRUE because..."
# Label: TRUE
# '''