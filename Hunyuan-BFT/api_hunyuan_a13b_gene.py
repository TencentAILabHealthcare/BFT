import os
from openai import OpenAI

# 本地 LLaMA-Factory API
BASE_URL = "http://127.0.0.1:8000/v1"
MODEL = "gpt-3.5-turbo"  # 若启动时设置了 API_MODEL_NAME，请改成对应名字
API_KEY = "EMPTY"
os.environ["OPENAI_API_KEY"] = API_KEY

def main():

    system = "\nYou are a helpful biology assistant.\n\n**IMPORTANT:** Keep your entire response concise—**the total number of characters (including spaces) must not exceed 2000**. \nIf the answer would be longer, please use brief sentences.\n"
    prompt = "tell me about gene CTSL."

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

# python /aaa/gelseywang/buddy1/lukatang/LLMs/BFT_github/Hunyuan-BFT/api_hunyuan_a13b_gene.py