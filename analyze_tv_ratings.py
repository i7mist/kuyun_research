import sys

import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from datetime import date, datetime
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

if len(sys.argv) > 1:
    check_day = int(sys.argv[1])
else:
    print("使用方法：python3 analyze_tv_ratings [新剧播出天数，首播日为0，第二天为1，以此类推]")
    exit(1)

with open("screen_history.pkl", "rb") as f:
    sh = pickle.load(f)

target_tv = "湖南卫视"
target_sh = {}
show_day_fmt = '%Y-%m-%d'
show_time_fmt = '%Y-%m-%d %H:%M'


def calc_show_prime_time_len(r, prime_time_begin, prime_time_end):
    start = max(datetime.strptime(r['start_time'], show_time_fmt), prime_time_begin)
    end = min(datetime.strptime(r['end_time'], show_time_fmt), prime_time_end)
    return end - start


all_show_seq = []
cur_show_seq = []
cur_show = None

for cur_date, data in sorted(sh.items()):
    if cur_date < date(2016, 1, 1):
        continue
    prime_time_begin = datetime.strptime(f'{cur_date} 19:00', show_time_fmt)
    prime_time_end = datetime.strptime(f'{cur_date} 22:00', show_time_fmt)
    if data["data"]:
        l = data["data"]["list"]
    else:
        l = []
    subl = []
    for r in l:
        if r["tv_name"] == target_tv:
            subl.append(r)
    if len(subl) == 0:
        # print(date, subl)
        continue
    elif len(subl) > 1:
        max_prime_len = None
        prime_show = None
        for r in subl:
            prime_len = calc_show_prime_time_len(r, prime_time_begin, prime_time_end)
            if not max_prime_len or prime_len > max_prime_len:
                max_prime_len = prime_len
                prime_show = r
    else:
        prime_show = subl[0]
    ca_name = prime_show['ca_name']
    tv_ratings = prime_show['tv_ratings']
    ep_list = prime_show['epg_name'].lstrip(ca_name + " ")
    if ep_list or ca_name == "红星照耀中国":
        eps = ep_list.split(" ")
        cur_show_r = {'ca_name': ca_name, 'tv_ratings': tv_ratings, 'eps': eps, 'year': prime_time_begin.year}
        if ca_name == cur_show or ca_name == "美味奇缘之二":
            cur_show_seq.append(cur_show_r)
        elif ca_name == "红星照耀中国" or eps[0] == '1' or not cur_show:
            if cur_show_seq:
                all_show_seq.append(cur_show_seq)
            cur_show = ca_name
            cur_show_seq = [cur_show_r]

same_eq_count = 0
dep_rel_dict_list = []

print(all_show_seq)

for i in range(1, len(all_show_seq)):
    cur_show_seq = all_show_seq[i]
    prev_show_seq = all_show_seq[i - 1]
    same_eq_count += 1
    dep_rel_dict_list.append({"cur_ca_name": cur_show_seq[check_day]['ca_name'],
                              "year": cur_show_seq[check_day]["year"],
                              "prev_ca_name": prev_show_seq[-1]['ca_name'],
                              "cur_rating": cur_show_seq[check_day]['tv_ratings'],
                              "prev_rating": prev_show_seq[-1]['tv_ratings'],
                              "cur_prev_ratio": cur_show_seq[check_day]['tv_ratings'] / prev_show_seq[-1][
                                  'tv_ratings'],
                              "prev_ep_count": len(prev_show_seq[-1]["eps"]),
                              "cur_ep_count": len(cur_show_seq[check_day]["eps"])})

dep_rel_df = pd.DataFrame(dep_rel_dict_list)
print(dep_rel_df)
sns.scatterplot(data=dep_rel_df, x='prev_rating', y='cur_rating')
plt.show()
mod = smf.ols(formula='cur_rating ~ prev_rating + year + prev_ep_count + cur_ep_count', data=dep_rel_df)
res = mod.fit()
print(res.summary())

# calculate variance inflation factor to quantify multicollinearity
X = add_constant(dep_rel_df[["prev_rating", "year", "prev_ep_count", "cur_ep_count"]])
print(pd.Series([variance_inflation_factor(X.values, i)
               for i in range(X.shape[1])],
              index=X.columns))
