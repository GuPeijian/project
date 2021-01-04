import torch
import sys
import json
sys.path.append("..")
from src.transformers  import BertTokenizer
from UniLM import FuseModel




model_path="/home/gpj/project/all_model/keywords_model_epoch2"

tokenizer = BertTokenizer.from_pretrained(model_path)

input_text="【朝中社：#朝鲜开城出现新冠疑似病例# 金正恩宣布该地区实施紧急状态】朝中社：朝鲜开城市7月19日发现一例新冠疑似病例。该名疑似患者三年前逃至韩国，7月19日非法越过朝韩边境返回朝鲜。朝鲜劳动党中央委员会政治局7月25日在党中央本部大楼召开了紧急扩大会议。朝鲜劳动党委员长、国务委员会委员长"
input_text2="【#日本研究发现变异新冠病毒#，#日本新增确诊多为变异新冠病毒感染者#】新华社消息，截至当地时间8月8日20时，日本当日新增新冠确诊病例1565例，连续两天单日新增病例超过1500例。据日媒报道，日本国立感染症研究所最新研究发现，6月以来在日本扩散的新冠病毒是变异后的、具有新型基因序列的新冠病毒。研究称，今年3月起日本疫情扩大，主要是由欧洲相关基因序列的新冠病毒导致，但在5月下旬已暂时平息。6月中旬起，以东京为中心出现了具有新型基因序列的新冠病毒，并向全国各地扩散。目前日本国内大量增加的确诊患者大都属于这种变异后新冠病毒的感染者。"
input_text3='【国药集团董事长：#国产新冠疫苗预计12月底上市# 两针不到一千元】“我打了两针新冠肺炎疫苗，没啥不良反应。”中国医药集团有限公司党委书记、董事长刘敬桢说。日前，国药集团中国生物北京生物制品研究所新冠灭活疫苗生产车间通过国家相关部门组织的生物安全联合检查，具备了使用条件。\n\n刘敬桢表示，国际临床三期试验结束后，灭活疫苗就可以进入审批环节，预计今年12月底能够上市。预计北京生物制品研究所的灭活疫苗年产量能达1.2亿剂，武汉生物制品研究所的灭活疫苗年产量能达1亿剂。\n\n刘敬桢称，“灭活疫苗上市后，价格不会很高，预计几百块钱一针。如果打两针的话，价格应在1000块钱以内。”“我国14亿人不是人人都有必要打，比如居住在人口密集城市的学生、上班族等是有必要的，而居住在人口稀少的农村地区的人们就可以不用打。”（光明日报）'
input_text4="【中疾控原副主任杨功焕：#全球疫情第二次发作已拉开序幕#】随着北半球气温转凉，秋冬降临，欧洲多个国家单日新增病例持续创新高。在亚洲，印度、尼泊尔等国疫情让人担忧，而美国单日新增病例则一直在高位徘徊不下。中国疾控中心原副主任杨功焕11日在接受记者采访时认为，综合多方面因素，新冠疫情秋冬季反弹已经开始，第二波疫情的发作已拉开序幕。不过她也认为，随着人类应对新冠疫情经验的不断积累，死亡率已在下降，所以不必像第一波疫情出现时那么担忧。"
input_text5="【青岛卫健委：#青岛已排查密接132人#，#青岛已采样277968份进行核酸检测#】据青岛卫健委，2020年10月11日我市发现3例新冠肺炎无症状感染者后，立即组织开展大规模的流调排查和分类检测，已采样277968份进行核酸检测。截至10月12日12时，全市已排查到密切接触者132人，全部实行集中隔离观察，全部完成核酸检测，其中9人核酸检测结果阳性，其余均为阴性；密切接触者的密切接触者171人，全部实行集中隔离观察，核酸检测结果均为阴性；一般接触者840人，核酸检测结果均为阴性；医务人员、住院病人及陪护人员162601人，已完成核酸检测154815人，结果均为阴性；社区检测人群114221人，已完成核酸检测75296人，结果均为阴性"
input_text6="【#中国成功发射高分十三号卫星# 用于国土普查等领域】北京时间10月12日0时57分，中国在西昌卫星发射中心用长征三号乙运载火箭，成功将高分十三号卫星发射升空，卫星顺利进入预定轨道。高分十三号卫星是高轨光学遥感卫星，主要用于国土普查、农作物估产、环境治理、气象预警预报和综合防灾减灾等领域，可为国民经济发展提供信息服务。"
input_text7="拜托，各地政府。据我了解，现在对香港来的人是否隔离以及怎么隔离，权限完全在地方政府的手里，对集中隔离扩展到多少国家和地区，决定权也在各地方政府的手里。不要再犹豫了，及时把集中隔离的措施扩大到中国大陆境外所有国家和地区来的人，现在已经到了必须这样做的时候。",
text_id=tokenizer.encode(input_text6,max_length=300,truncation=True,add_special_tokens=True)

user_ids= [tokenizer.convert_tokens_to_ids('<user1>'), tokenizer.convert_tokens_to_ids('<user2>')]
keywords_id=tokenizer.convert_tokens_to_ids('<keywords>')
keywords=["中国","超越"]
text_id+= [keywords_id]
for keyword in keywords:
    keyword_id = tokenizer.encode(keyword, add_special_tokens=False)
    keyword_id += [tokenizer.sep_token_id]
    text_id+=keyword_id
input_ids=text_id+[user_ids[0]]
input_ids=torch.tensor([input_ids])
text_len=input_ids.size()[1]
device = torch.device("cuda")
input_ids=input_ids.to(device)
#sentiment=torch.tensor([[2]]).to(device)

model=FuseModel.from_pretrained(model_path)
model.to(device)
model.eval()

times=10

c=[]

for i in range(times):
    a=model.generate(input_ids,topp=0.96,no_repeat_ngram_size=3,do_sample=True,max_length=(text_len+50),min_length=(text_len+6),
                     early_stopping=True,temperature=0.5,sentiment_input=None)
    c.append(tokenizer.decode(a[0][input_ids.size()[1]:]))

print(c)
# file="./comment_output_new/test2.json"
# with open(file,'w',encoding='utf-8')as f:
#     json.dump(c,f,ensure_ascii=False,indent=4)


    
    


