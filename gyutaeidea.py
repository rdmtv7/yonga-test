import pandas as pd
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pprint

nltk.download("wordnet")
nltk.download("stopwords")
nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")

lemmatizer = WordNetLemmatizer()
from collections import Counter

df = pd.read_excel("이노싯톨.xlsx")

data = df[["Keywords", "Search Volume"]].dropna()
data.rename(
    columns={"Keywords": "search_query", "Search Volume": "search_volume"}, inplace=True
)

data["kw_split"] = data.search_query.apply(lambda x: word_tokenize(x))
data["kw_split"] = data.kw_split.apply(
    lambda x: [kw for kw in x if any(char.isalpha() or char.isdigit() for char in kw)]
)  # split 된 키워드 중에서 '%', '&' 와 같이 단독으로 기호만 쓰이는 것들은 제외시킴

exclude_words = ["pms", "another_word", "etc"]  # nomalization을 Skip할 키워드 리스트, 예시 : PMS
data["kw_split_lemma"] = data.kw_split.apply(
    lambda x: [
        kw
        if pos_tag([kw])[0][1] == "NN" and kw not in exclude_words
        else lemmatizer.lemmatize(kw)
        for kw in x
    ]
)  # exclude_words에 들어있지 않은 키워드 중에서, pms처럼 고유명사인 경우엔 nomalization에서 제외시킴


def perform_analysis_with_stopwords_removal(data, tier1_num, tier2_num, stopwords_list):
    # 이미 계산된 root word 빈도와 search volume을 사용, 불용어 제거
    root_word_freq = Counter(
        [
            word
            for sublist in data["kw_split_lemma"]
            for word in sublist
            if word not in stopwords_list
        ]
    )
    root_word_volumes = {}
    for root_word in root_word_freq.keys():
        total_volume = data[
            data["kw_split_lemma"].apply(
                lambda x: root_word in x and root_word not in stopwords_list
            )
        ]["search_volume"].sum()
        root_word_volumes[root_word] = total_volume
    sorted_root_word_volumes = sorted(
        root_word_volumes.items(), key=lambda x: x[1], reverse=True
    )

    print(sorted_root_word_volumes)
    # 결과를 저장할 리스트
    results = []

    # tier1 반복
    for tier1_idx, (tier1, tier1_volume) in enumerate(sorted_root_word_volumes):
        if tier1_idx >= tier1_num:
            break

        # tier1을 포함하는 데이터
        tier1_data = data[
            data["kw_split_lemma"].apply(
                lambda x: tier1 in x and tier1 not in stopwords_list
            )
        ]

        # tier2 계산
        tier2_root_word_freq = Counter(
            [
                word
                for sublist in tier1_data["kw_split_lemma"]
                for word in sublist
                if word not in stopwords_list
            ]
        )
        tier2_root_word_volumes = {}
        for root_word in tier2_root_word_freq.keys():
            total_volume = tier1_data[
                tier1_data["kw_split_lemma"].apply(
                    lambda x: root_word in x and root_word not in stopwords_list
                )
            ]["search_volume"].sum()
            tier2_root_word_volumes[root_word] = total_volume
        sorted_tier2_root_word_volumes = sorted(
            tier2_root_word_volumes.items(), key=lambda x: x[1], reverse=True
        )[:tier2_num]

        # tier2_window 생성
        tier2_window = [tier2[0] for tier2 in sorted_tier2_root_word_volumes]

        # 결과 저장
        results.append((tier1, tier1_volume, tier2_window))

        # tier1 및 tier2를 전체 목록에서 제거
        sorted_root_word_volumes = [
            item
            for item in sorted_root_word_volumes
            if item[0] not in [tier1] + tier2_window
        ]

    return results


# 이 함수는 'data', 'tier1_num', 'tier2_num', 그리고 'stopwords_list'를 인자로 필요로 합니다.
# 'stopwords_list'는 제거하고자 하는 불용어 리스트입니다.


# 이 함수는 'data', 'tier1_num', 'tier2_num' 인자를 필요로 합니다.
# 'data'는 앞서 정의한 DataFrame이어야 하고, 'tier1_num'과 'tier2_num'은 반복 횟수를 지정하는 숫자입니다.

tier1_num = int(input("Tier 1 ROOT 갯수를 입력하세요 ~_~ : "))
tier2_num = int(input("Tier 2 ROOT 갯수를 입력하세요 ~_~ : "))
print(
    perform_analysis_with_stopwords_removal(
        data, tier1_num, tier2_num, stopwords.words("english") + ["para", "for"]
    )
)
