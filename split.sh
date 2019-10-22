# 1-1821 共1082个
ls |  head -1821 |xargs -i cp -r {} ../train/1
# 1822 - 2428 共607个
ls | tail -n +1822 | head -n 607 |xargs -i cp -r {} ../validation/1 

# 复制 1821 之后的607 个 到验证集
# 包括1821行 

# 2429-3035 共607个
ls | tail -n +2429 | head -n 607 |xargs  -i cp -r {} ../test/1
