import os, requests, bs4
import pandas as pd

url = 'https://www.tutorialspoint.com/java/java_interview_questions.htm'

#Getting the webpage
print('Getting page %s...' % url)
res = requests.get(url)
res.raise_for_status()
soup = bs4.BeautifulSoup(res.text, "lxml")

#Getting image url
questions = soup.select('.toggle')
answers = soup.select('.toggle-content')

question_text = ""
answer_text = ""

columns = ["question", "answer"]
df = pd.DataFrame(columns=columns)

curr_id = 0
for i in range(len(questions)):
    df.loc[curr_id] = [questions[i].select('label')[0].text, answers[i].select('p')[0].text]
    curr_id += 1

df.to_csv("qna.csv")