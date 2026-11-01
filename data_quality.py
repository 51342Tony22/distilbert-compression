# 用于检查并处理缺失值重复数据和标签冲突

#print(df.head()) //读取表头，前五行
#print(df.shape)    //有多少行多少列
#print(df.columns)  //查看各字段名称和数据类型

#print(df.isnull().sum()) //检查空值
#print(df["category"].value_counts()) //类别数量，每类多少个
#print(df.duplicated().sum()) //检查整行重复
#print(df["text"].value_counts()) //只看text有多少重复


#######################################
# 第二周更新内容
#######################################
def check_missing_values(df):
    return df.isnull().sum()

def handle_missing_values(df):
    before = len(df)
    df = df.dropna(subset=["text", "category"]).copy()
    after = len(df)
    print(f"Removed missing rows: {before - after}")
    return df

def check_label_conflicts(df):
    conflicts = (df.groupby("text")["category"].nunique())
    conflicts = conflicts[conflicts > 1]
    return conflicts

def remove_label_conflicts(df):
    conflict_texts = (df.groupby("text")["category"].nunique())
    conflict_texts = conflict_texts[conflict_texts > 1].index

    before = len(df)
    df = df[~df["text"].isin(conflict_texts)].copy()
    after = len(df)

    print(f"Removed label conflict rows: {before - after}")
    return df

def category_distribution(df):

    return (
        df["category"]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
    )

def check_duplicates(df):
    return df.duplicated().sum()

######################################

#print("原始数据形状：")
#print(df.shape)

#df_unique = df.drop_duplicates()  #去重

#print("去重后的数据形状：")
#print(df_unique.shape)

#print("去重后的类别分布：")
#print(df_unique["category"].value_counts())

#conflict_check = df.groupby("text")["category"].nunique()
#print(conflict_check[conflict_check > 1])

# 350 行、2 列
# 无缺失值
#- problem_consultation: 220
#- complaint_suggestion: 84
#- feature_feedback: 46
