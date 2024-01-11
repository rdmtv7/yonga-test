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

df = pd.read_csv('testfile.csv')

data = df[['Normalized Root', 'Broad Search Volume']].copy()
data.rename(columns = {'Normalized Root' : 'search_query', 'Broad Search Volume' : 'search_volume'}, inplace = True)

data['kw_split'] = data.search_query.apply(lambda x : word_tokenize(x))
data['kw_split'] = data.kw_split.apply(lambda x : [kw for kw in x if any(char.isalpha() or char.isdigit() for char in kw)]) # split 된 키워드 중에서 '%', '&' 와 같이 단독으로 기호만 쓰이는 것들은 제외시킴

exclude_words = ['pms','another_word', 'etc'] # nomalization을 Skip할 키워드 리스트, 예시 : PMS
data['kw_split_lemma'] = data.kw_split.apply(lambda x: [kw if pos_tag([kw])[0][1] == 'NN' and kw not in exclude_words else lemmatizer.lemmatize(kw) for kw in x]) # exclude_words에 들어있지 않은 키워드 중에서, pms처럼 고유명사인 경우엔 nomalization에서 제외시킴

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

  sq_count = dict(sorted(sq_count.items(), key = lambda x : x[1], reverse = True))

  stop_words = set(stopwords.words('english'))
  for key in stop_words:
      if key in sq_count.keys():
          del sq_count[key]

  tier1_kw, _ = sorted(sq_count.items(), key = lambda x : x[1], reverse = True)[0]
  data['remaining_roots'] = data.kw_split_lemma.apply(lambda x : [kw for kw in x if tier1_kw not in kw])
  tier1_rows = data[data.search_query.str.contains(tier1_kw)]
  remain_kws = tier1_rows.remaining_roots.tolist()
  total_tier2_kws = list(set(sum(remain_kws, [])))

  tier2_sq_count = {}
  for kw in total_tier2_kws:
      sum_sv = tier1_rows[tier1_rows.remaining_roots.apply(lambda x : kw in x)]['search_volume'].sum()
      tier2_sq_count[kw] = sum_sv

  tier2_sq_count = dict(sorted(tier2_sq_count.items(), key = lambda x : x[1], reverse = True))
  tier2_kws = [kw for kw, _ in tier2_sq_count.items()][:10]

  sub_dict = {}
  for kw in tier2_kws:
      sub_df = tier1_rows[tier1_rows.search_query.apply(lambda x : kw in x)]
      sub_dict[kw] = sub_df
  sub_dict['uncategorized'] = tier1_rows[~tier1_rows.remaining_roots.apply(lambda x : any(kw in x for kw in tier2_kws))]

  data = data[~data.search_query.apply(lambda x : any(kw in x for kw in [tier1_kw]))]
  return tier1_kw, sub_dict


if __name__ == '__main__':

  tier1_num = int(input('Tier 1 ROOT 갯수를 입력하세요 ~_~ : '))

  cnt = 0
  total_kw_group = {}
  group_check = []

  while True:
      cnt += 1
      tier1_kw, tier2_sub_dict = classify_kws()
      total_kw_group[tier1_kw] = tier2_sub_dict

      if cnt == tier1_num:
          break

  print(f'Tier 1 키워드 >> {list(total_kw_group.keys())}')
  for kw in total_kw_group.keys():
    print(f'{kw} >> {list(total_kw_group[kw].keys())}')
    