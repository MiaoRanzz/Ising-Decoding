API = "qk_CoL31u9Fe9bn33Ibm8cajbDgG0eeZKBf_uhkANQG7IY"

from lqcloud import save_account
save_account(
    api_key=API,
    url="https://cloud.logicalqubit.com"
)   # 信息保存在 ~/.lqcloud/config.json