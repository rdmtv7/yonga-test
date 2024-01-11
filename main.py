import pandas as pd
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pprint

nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

lemmatizer = WordNetLemmatizer()

df = pd.read_excel('이노시톨.xlsx')

data = df[['Keywords', 'Search Volume']].dropna()
data.rename(columns={
    'Keywords': 'search_query',
    'Search Volume': 'search_volume'
},
            inplace=True)

data['kw_split'] = data.search_query.apply(lambda x: word_tokenize(x))
data['kw_split'] = data.kw_split.apply(lambda x: [
    kw for kw in x if any(char.isalpha() or char.isdigit() for char in kw)
])  # split 된 키워드 중에서 '%', '&' 와 같이 단독으로 기호만 쓰이는 것들은 제외시킴

exclude_words = ['pms', 'another_word',
                 'etc']  # nomalization을 Skip할 키워드 리스트, 예시 : PMS
data['kw_split_lemma'] = data.kw_split.apply(lambda x: [
    kw if pos_tag([kw])[0][1] == 'NN' and kw not in exclude_words else
    lemmatizer.lemmatize(kw) for kw in x
])  # exclude_words에 들어있지 않은 키워드 중에서, pms처럼 고유명사인 경우엔 nomalization에서 제외시킴


def classify_kws():
  global data
  sq_count = {}
  for row in data.itertuples():
    volume = row.search_volume
    kw_list = row.kw_split_lemma

    for kw in kw_list:
      if kw in sq_count.keys():
        sq_count[kw] = sq_count[kw] + volume
      else:
        sq_count[kw] = volume

  stop_words = set(stopwords.words('english') + ['para', 'for'])
  for key in stop_words:
    if key in sq_count.keys():
      del sq_count[key]
  # 'Tier 1' 키워드 식별
  tier1_kw, tier1_count = sorted(sq_count.items(),
                                 key=lambda x: x[1],
                                 reverse=True)[0]

  # 'Tier 2' 키워드 및 검색 볼륨 집계
  tier2_sq_count = {}
  tier1_rows = data[data['kw_split_lemma'].apply(lambda x: tier1_kw in x)]
  for row in tier1_rows.itertuples():
    volume = row.search_volume
    kw_list = row.kw_split_lemma

    for kw in kw_list:
      if kw != tier1_kw:  # 'Tier 1' 키워드는 제외
        if kw in tier2_sq_count.keys():
          tier2_sq_count[kw] += volume
        else:
          tier2_sq_count[kw] = volume

  for key in stop_words:
    if key in tier2_sq_count.keys():
      del tier2_sq_count[key]

  # 'Tier 2' 키워드 검색 볼륨에 따른 정렬
  sorted_tier2_kws = sorted(tier2_sq_count.items(),
                            key=lambda x: x[1],
                            reverse=True)
  tier2_kws = [kw for kw, _ in sorted_tier2_kws]

  # 데이터 필터링
  data = data[~data['kw_split_lemma'].
              apply(lambda x: any(kw in x for kw in [tier1_kw]))]

  return (tier1_kw, tier1_count), sorted_tier2_kws


if __name__ == '__main__':

  tier1_num = int(input('Tier 1 ROOT 갯수를 입력하세요 ~_~ : '))

  cnt = 0
  total_kw_group = {}
  group_check = []
  while True:
    cnt += 1
    tier1_kw, tier2_kws = classify_kws()
    total_kw_group[tier1_kw] = tier2_kws
    group_check.append([tier1_kw, tier2_kws])
    if cnt == tier1_num:
      break

  pprint.pprint(group_check)
