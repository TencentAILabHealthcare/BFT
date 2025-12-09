import os
from openai import OpenAI

# 本地 LLaMA-Factory API
BASE_URL = "http://127.0.0.1:8000/v1"
MODEL = "gpt-3.5-turbo"  # 若启动时设置了 API_MODEL_NAME，请改成对应名字
API_KEY = "EMPTY"
os.environ["OPENAI_API_KEY"] = API_KEY

def main():
    system = "You are an efficient and insightful assistant to a molecular biologist."
    task = lambda genes: f"Your task is to propose a biological process term for gene sets. Here is the gene set: {genes}"
    chain = """
    Let do the task step-by-step:
    Step1, write a cirtical analysis for gene functions. For each important point, discribe your reasoning and supporting information.
    Step2, analyze the functional associations among different genes from the critical analysis.
    Step3, summarize a brief name for the most significant biological process of gene set from the functional associations. 
    """
    instruction = """
    Put the name at the top of analysis as "Process: <name>" and follow the analysis.
    Be concise, do not use unnecessary words.
    Be specific, avoid overly general statements such as "the proteins are involved in various cellular processes".
    Be factual, do not editorialize.
    """

    # 单个样本（示例基因集）
    genes = "ZMPSTE24, BANF1, WRN, LMNA"
    prompt = task(genes) + chain + instruction

    print("User prompt:\n")
    print(prompt)
    print("\nAnswer:\n")

    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=2048,
        )
        print(resp.choices[0].message.content.strip())
    except Exception as e:
        print(f"Error during generation: {e}")

if __name__ == "__main__":
    main()

# python /aaa/gelseywang/buddy1/lukatang/LLMs/BFT_github/Hunyuan-BFT/api_hunyuan_a13b_pathway.py