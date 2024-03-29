import requests
import json
PyTorch_REST_API_URL = 'http://127.0.0.1:2000/generation'

text="【#中国成功发射高分十三号卫星# 用于国土普查等领域】北京时间10月12日0时57分，中国在西昌卫星发射中心用长征三号乙运载火箭，成功将高分十三号卫星发射升空，卫星顺利进入预定轨道。高分十三号卫星是高轨光学遥感卫星，主要用于国土普查、农作物估产、环境治理、气象预警预报和综合防灾减灾等领域，可为国民经济发展提供信息服务。"
text2="【青岛卫健委：#青岛已排查密接132人#，#青岛已采样277968份进行核酸检测#】据青岛卫健委，2020年10月11日我市发现3例新冠肺炎无症状感染者后，立即组织开展大规模的流调排查和分类检测，已采样277968份进行核酸检测。截至10月12日12时，全市已排查到密切接触者132人，全部实行集中隔离观察，全部完成核酸检测，其中9人核酸检测结果阳性，其余均为阴性；密切接触者的密切接触者171人，全部实行集中隔离观察，核酸检测结果均为阴性；一般接触者840人，核酸检测结果均为阴性；医务人员、住院病人及陪护人员162601人，已完成核酸检测154815人，结果均为阴性；社区检测人群114221人，已完成核酸检测75296人，结果均为阴性"
text3="美英都跳出来支持澳大利亚，他们给中国贴“战狼外交”标签，想封住中国外交官的嘴。但还有老胡呢。乌合麒麟昨晚发了新作，老胡于夜里将那幅新作贴到了我的个人推特上，到目前已获1300多次转发，5600多个点赞，绝对是个热推。老胡可是新闻自由，美英澳拿我没有办法。 近日中澳冲突中，环球时报和老胡自然处在子弹落下来最密集的前沿位置。我们不怕，我们经打，在西方舆论场上连续保持了环球时报声音的存在。这场冲突当然不仅是中国与澳大利亚之间的问题，它同时是“五眼联盟”对中国的舆论战。环球时报和老胡不会缺席的，我们永远与中国国家利益在一起。"
texts=[]
texts.append(text)
mid=1
history=["中国加油，中国航天加油"]
def predict_result(texts,history):
    payload={"text":texts,"label":"call","mid":mid,"history_generation":history,}
    r=requests.post(PyTorch_REST_API_URL,json=payload).json()

    if r["success"]:
        print(r["comment"])
    else:
        print("Request failed")


predict_result(texts,history)