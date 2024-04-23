import random 
# 生成随机数
def random_score():
    return random.randint(1,100)
# 级别判定
def get_rank(score):
    if score>= 90 :
        return 'A'
    elif score>= 80 :
        return 'B'
    else :
        return 'C'

# main 
for i in range(20):
    print(get_rank(random_score()) ,end =" ")