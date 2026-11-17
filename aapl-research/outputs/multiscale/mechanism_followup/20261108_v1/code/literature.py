from common import *
import urllib.request,urllib.parse,concurrent.futures
from datetime import datetime,timezone

REFERENCES=[('xpatch','10.1609/aaai.v39i19.34270'),('dlinear','10.1609/aaai.v37i9.26317'),
 ('mixture','10.1162/neco.1991.3.1.79'),('combination','10.1057/jors.1969.103'),
 ('shrinkage','10.1093/rfs/hhm055'),('lstm','10.1162/neco.1997.9.8.1735')]

def fetch(item):
    name,doi=item;url='https://api.crossref.org/works/'+urllib.parse.quote(doi,safe='')
    try:
        req=urllib.request.Request(url,headers={'User-Agent':'Academic-metadata-verification/1.0'})
        with urllib.request.urlopen(req,timeout=30) as r:body=r.read();status=r.status
        path=BASE/'literature'/f'{name}_crossref.json';path.write_bytes(body);m=json.loads(body)['message']
        return {'name':name,'doi':doi,'url':url,'http_status':status,'sha256':sha(path),
                'title':m.get('title'),'authors':m.get('author'),'issued':m.get('issued'),'container':m.get('container-title')}
    except Exception as e:return {'name':name,'doi':doi,'url':url,'error':repr(e)}

def main():
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:results=list(pool.map(fetch,REFERENCES))
    # TCN is an arXiv report; retain the actual source response separately.
    url='https://arxiv.org/abs/1803.01271'
    try:
        with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'}),timeout=30) as r:body=r.read();status=r.status
        path=BASE/'literature/tcn_arxiv.html';path.write_bytes(body)
        results.append({'name':'tcn','url':url,'http_status':status,'sha256':sha(path)})
    except Exception as e:results.append({'name':'tcn','url':url,'error':repr(e)})
    atomic(BASE/'literature/SOURCE_EVIDENCE.json',{'at_utc':datetime.now(timezone.utc).isoformat(),'sources':results,
        'scope':'Source metadata and retained page evidence; not a systematic novelty search or full-text review'})
    (BASE/'literature/LITERATURE_DIFFERENCES.md').write_text('''# 直接相关文献与候选的差异

| 先例 | 已有内容 | 本研究可检验的具体区别/所需对照 |
| --- | --- | --- |
| xPatch, Stitsyuk & Choi (2025) | 指数平滑、趋势/残差双流、分块非线性预测 | 在相同 AAPL 历史规则中检验锚点和分支；候选用小型因果卷积与显式基准收缩，不能将双流本身认作创新 |
| DLinear, Zeng et al. (2023) | 简单分解与线性预测可与复杂架构竞争 | 必须保留趋势单路及相同派生特征的线性对照，不能以参数少直接主张贡献 |
| Jacobs et al. (1991) | 专家混合及输入相关门控 | 状态混合不是新概念；用固定混合、单路和无收缩对照判断在本任务中是否必要 |
| Bates & Granger (1969) | 预测组合 | 不同专家不自动提供可重复的互补信号；保留简单组合控制 |
| Campbell & Thompson (2008) | 财务收益预测的样本外评价、受约束预测与弱信号问题 | 保留零基准和简单收缩控制；不将股票市场总体收益的结论直接迁移到 AAPL 分钟收益 |
| Bai, Kolter & Koltun (2018) | 因果/膨胀时序卷积 | 两层因果卷积只是实现选择；读出、模型大小和信息集需披露 |
| Hochreiter & Schmidhuber (1997) | LSTM 记忆机制 | 本文只检验所实现的最后状态/均值读出配方，不能泛化为 LSTM 理论能力判断 |

本研究拟贡献为针对性诊断、可复现的小模型对照和明确的失败证据。上述组件及其简单拼接不单独构成新颖性证明；结论中不使用“首个”“全新架构”或普适优越性主张。来源身份以 SOURCE_EVIDENCE.json 中实际获取结果为准；元数据获取不等于已完成全文系统综述。
''')
    print(json.dumps({'sources':len(results),'retrieved':sum('http_status' in r for r in results)}))

if __name__=='__main__':main()
