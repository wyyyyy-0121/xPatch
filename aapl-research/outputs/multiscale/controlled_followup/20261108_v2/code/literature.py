from common import *
import urllib.request,urllib.parse,re,html,concurrent.futures,shutil

SOURCES=[('tcn','https://arxiv.org/abs/1803.01271'),
    ('combination','https://api.crossref.org/works/10.1057%2Fjors.1969.103'),
    ('mixture','https://api.crossref.org/works/10.1162%2Fneco.1991.3.1.79'),
    ('shrinkage','https://api.crossref.org/works/10.1093%2Frfs%2Fhhm055')]

def fetch(item):
    name,url=item
    try:
        req=urllib.request.Request(url,headers={'User-Agent':'Academic-research-source-check/1.0'})
        with urllib.request.urlopen(req,timeout=25) as r:body=r.read();status=r.status
        p=BASE/'literature'/f'{name}.source';p.write_bytes(body)
        if name=='tcn':
            text=html.unescape(re.sub('<[^>]+>',' ',body.decode(errors='replace')))
            (BASE/'literature'/f'{name}.txt').write_text(text)
        else:
            data=json.loads(body)['message'];text=json.dumps({k:data.get(k) for k in ('title','author','abstract','issued','container-title')},ensure_ascii=False,indent=2)
            (BASE/'literature'/f'{name}.txt').write_text(text)
        return {'name':name,'url':url,'status':status,'sha256':sha(p),'evidence':'retained source page/metadata; not a systematic full-text novelty review'}
    except Exception as e:return {'name':name,'url':url,'error':repr(e)}

def main():
    old=PREVIOUS/'literature'
    if not (BASE/'literature/previous_stage_sources').exists():shutil.copytree(old,BASE/'literature/previous_stage_sources')
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:out=list(pool.map(fetch,SOURCES))
    atomic(BASE/'literature/SOURCE_EVIDENCE.json',{'sources':out,'original_core_references_retained':True})
    (BASE/'literature/LITERATURE_DIFFERENCES_ZH.md').write_text('''# 文献差异及本研究可主张的范围

膨胀因果卷积：Bai、Kolter、Koltun (2018) 已有。八层卷积及膨胀率不是新架构贡献；本研究检验固定小参数量下，特定 AAPL 配方的历史跨度与损失因素。

预测组合：Bates 与 Granger (1969) 已有。专家不同不能保证互补；本研究用固定输出、零专家、等权/常数及按时间校准的状态权重检验增量。

门控专家混合：Jacobs 等 (1991) 已有。四状态特征到 softmax 权重不是新颖性本身；必须与常数权重比较，并披露整个四专家系统成本。

受约束收益预测：Campbell 与 Thompson (2008) 讨论市场总体收益；不能把结论直接迁移到 AAPL 分钟收益。零预测及内层缩放控制检验当前数据下的弱信号边界。

xPatch、PatchTST、TFT、AAPL 多期限预测、时间切分与块重采样等先例沿用原主稿引用。原数据定义、公司行动及全任务对照继续是论文主体。方法贡献须由实际受控结果支持，不能仅以拼接既有组件或训练完成作为创新。

本次保留源页面/元数据并检查题名、作者和适用主题；尚不声称完成系统综述。来源请求失败保留错误，不凭空补全获取记录。
''',encoding='utf-8')
    print(json.dumps(out),flush=True)

if __name__=='__main__':main()
