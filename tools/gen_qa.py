# -*- coding: utf-8 -*-
"""生成 QA 评测集（200+ 条）。

设计要点：
  1. 每题标注 gold_doc，判定口径为文档级（命中该 doc 记 1）
  2. 刻意使用与文档不同的措辞（"出差住酒店的标准" vs 文档里的"住宿标准"），
     否则字面匹配就能刷到满分，评测失去意义
  3. 每个查询的权限组都包含 public，因此候选集里必然混入公开文档作为干扰项
  4. 末尾 50 条是改写题（同一语义换说法），专门测语义召回能力
"""

from __future__ import annotations

import json
from pathlib import Path

# (question, tenant, groups, gold_doc)
QA: list[tuple[str, str, str, str]] = [
    # ---------------- acme-handbook ----------------
    ("员工手册适用于哪些人？", "acme", "public", "acme-handbook"),
    ("入职后多久之内要激活内部系统账号？", "acme", "public", "acme-handbook"),
    ("连续旷工多久公司可以解除劳动合同？", "acme", "public", "acme-handbook"),
    ("工作满十年年假有几天？", "acme", "public", "acme-handbook"),
    ("年假跨年没休完怎么处理？", "acme", "public", "acme-handbook"),
    ("公司规定几点开始上班？", "acme", "public", "acme-handbook"),
    ("请病假需要提供什么材料？", "acme", "public", "acme-handbook"),
    ("员工手册修订后怎么生效？", "acme", "public", "acme-handbook"),

    # ---------------- acme-leave ----------------
    ("弹性工作制最晚几点到岗？", "acme", "public", "acme-leave"),
    ("忘记打卡一个月最多能补几次？", "acme", "public", "acme-leave"),
    ("休息日加班按几倍计算？", "acme", "public", "acme-leave"),
    ("加班换的调休多久之内要用完？", "acme", "public", "acme-leave"),
    ("一年最多能请多少天事假？", "acme", "public", "acme-leave"),
    ("婚假有几天？", "acme", "public", "acme-leave"),
    ("一个月迟到几次算半天事假？", "acme", "public", "acme-leave"),
    ("什么情况会被认定为旷工？", "acme", "public", "acme-leave"),

    # ---------------- acme-conduct ----------------
    ("访客进入办公区要走什么流程？", "acme", "public", "acme-conduct"),
    ("门禁卡丢了应该怎么办？", "acme", "public", "acme-conduct"),
    ("会议纪要要在多久之内发出去？", "acme", "public", "acme-conduct"),
    ("办公区里禁止做哪些事？", "acme", "public", "acme-conduct"),
    ("跟客户沟通能不能用个人微信？", "acme", "public", "acme-conduct"),

    # ---------------- acme-se-baseline ----------------
    ("系统密码多久要更换一次？", "acme", "public", "acme-se-baseline"),
    ("口令长度最少要多少位？", "acme", "public", "acme-se-baseline"),
    ("屏幕锁定的等待时间不能超过多久？", "acme", "public", "acme-se-baseline"),
    ("收到冒充领导要求转账的邮件怎么办？", "acme", "public", "acme-se-baseline"),
    ("往外部邮箱发大附件有什么限制？", "acme", "public", "acme-se-baseline"),
    ("电脑中了病毒应该找谁处理？", "acme", "public", "acme-se-baseline"),

    # ---------------- acme-onboarding ----------------
    ("入职当天要带哪些材料？", "acme", "public", "acme-onboarding"),
    ("信息安全考试多少分算合格？", "acme", "public", "acme-onboarding"),
    ("试用期一般多长时间？", "acme", "public", "acme-onboarding"),
    ("转正评估包含哪几个维度？", "acme", "public", "acme-onboarding"),
    ("工资每个月几号发放？", "acme", "public", "acme-onboarding"),

    # ---------------- acme-welfare ----------------
    ("交通补贴一个月多少钱？", "acme", "public", "acme-welfare"),
    ("补充医疗保险家属能不能参加？", "acme", "public", "acme-welfare"),
    ("健康体检多久组织一次？", "acme", "public", "acme-welfare"),
    ("一线业务岗的通讯补贴是多少？", "acme", "public", "acme-welfare"),
    ("社保和公积金从什么时候开始缴？", "acme", "public", "acme-welfare"),

    # ---------------- acme-travel ----------------
    ("一线城市出差住宿每晚最多报销多少？", "acme", "public,finance", "acme-travel"),
    ("出差期间的市内交通费怎么算？", "acme", "public,finance", "acme-travel"),
    ("发票丢了还能报销吗？", "acme", "public,finance", "acme-travel"),
    ("费用发生后多久内要提交报销单？", "acme", "public,finance", "acme-travel"),
    ("两万元以上的报销要谁审批？", "acme", "public,finance", "acme-travel"),
    ("坐飞机要提前几个工作日申请？", "acme", "public,finance", "acme-travel"),
    ("哪些费用是明确不给报销的？", "acme", "public,finance", "acme-travel"),

    # ---------------- acme-budget ----------------
    ("次年预算编制什么时候启动？", "acme", "public,finance", "acme-budget"),
    ("季度预算执行率低于多少要写说明？", "acme", "public,finance", "acme-budget"),
    ("支出超过预算余额系统会怎么处理？", "acme", "public,finance", "acme-budget"),
    ("预算考核在绩效里占多少权重？", "acme", "public,finance", "acme-budget"),
    ("公司预算分为哪几类？", "acme", "public,finance", "acme-budget"),

    # ---------------- acme-procurement ----------------
    ("五万元以上的采购要走什么流程？", "acme", "public,finance", "acme-procurement"),
    ("付款申请要哪四单齐全？", "acme", "public,finance", "acme-procurement"),
    ("供应商的标准账期是多少天？", "acme", "public,finance", "acme-procurement"),
    ("预付款最多能付多少比例？", "acme", "public,finance", "acme-procurement"),
    ("引入新供应商要审核哪些内容？", "acme", "public,finance", "acme-procurement"),

    # ---------------- acme-invoice ----------------
    ("发票抬头写错了能不能报？", "acme", "public,finance", "acme-invoice"),
    ("发票查验要在多久之内完成？", "acme", "public,finance", "acme-invoice"),
    ("一千元以上的发票要附什么？", "acme", "public,finance", "acme-invoice"),
    ("发票开错了要怎么红冲？", "acme", "public,finance", "acme-invoice"),
    ("哪些发票财务一律不受理？", "acme", "public,finance", "acme-invoice"),

    # ---------------- acme-compensation ----------------
    ("公司一共设置了几个专业职级？", "acme", "public,hr", "acme-compensation"),
    ("绩效拿 A 最多能涨多少工资？", "acme", "public,hr", "acme-compensation"),
    ("年终奖的基数是几个月工资？", "acme", "public,hr", "acme-compensation"),
    ("绩效为 C 的员工有年终奖吗？", "acme", "public,hr", "acme-compensation"),
    ("入职不满半年的员工年终奖怎么算？", "acme", "public,hr", "acme-compensation"),
    ("职级晋升一年开展几次？", "acme", "public,hr", "acme-compensation"),
    ("薪酬信息能不能跟同事讨论？", "acme", "public,hr", "acme-compensation"),

    # ---------------- acme-performance ----------------
    ("绩效考核里业绩指标占多少权重？", "acme", "public,hr", "acme-performance"),
    ("A 档最多占部门人数的多少？", "acme", "public,hr", "acme-performance"),
    ("对考核结果有异议怎么申诉？", "acme", "public,hr", "acme-performance"),
    ("连续两年考核为 D 会怎么样？", "acme", "public,hr", "acme-performance"),
    ("行为指标包含哪几个维度？", "acme", "public,hr", "acme-performance"),

    # ---------------- acme-recruitment ----------------
    ("技术岗位面试一共有几轮？", "acme", "public,hr", "acme-recruitment"),
    ("背景调查主要核查什么？", "acme", "public,hr", "acme-recruitment"),
    ("接受了 offer 又反悔会怎样？", "acme", "public,hr", "acme-recruitment"),
    ("内部推荐成功能拿多少奖金？", "acme", "public,hr", "acme-recruitment"),
    ("面试官多久之内要提交评价？", "acme", "public,hr", "acme-recruitment"),

    # ---------------- acme-training ----------------
    ("公司每年提供不少于多少学时的培训？", "acme", "public,hr", "acme-training"),
    ("外部培训超过多少钱要分管副总批？", "acme", "public,hr", "acme-training"),
    ("培训费用超过一万要签什么协议？", "acme", "public,hr", "acme-training"),
    ("担任导师需要什么条件？", "acme", "public,hr", "acme-training"),
    ("职业发展有几条通道？", "acme", "public,hr", "acme-training"),

    # ---------------- acme-account ----------------
    ("员工离职后账号多久注销？", "acme", "public,it", "acme-account"),
    ("权限复核多久开展一次？", "acme", "public,it", "acme-account"),
    ("高权限账号多久复核一次？", "acme", "public,it", "acme-account"),
    ("使用特权账号操作要不要录屏？", "acme", "public,it", "acme-account"),
    ("审批人能不能审批自己发起的单据？", "acme", "public,it", "acme-account"),

    # ---------------- acme-ops ----------------
    ("P0 故障多久之内要响应？", "acme", "public,it", "acme-ops"),
    ("生产环境变更安排在什么时间？", "acme", "public,it", "acme-ops"),
    ("故障复盘要在多久之内完成？", "acme", "public,it", "acme-ops"),
    ("备份数据保留多久？", "acme", "public,it", "acme-ops"),
    ("恢复演练多久开展一次？", "acme", "public,it", "acme-ops"),

    # ---------------- acme-device ----------------
    ("固定资产一年盘点几次？", "acme", "public,it", "acme-device"),
    ("离职时设备没还会怎么样？", "acme", "public,it", "acme-device"),
    ("设备坏了能不能自己拆机？", "acme", "public,it", "acme-device"),
    ("归还的设备数据怎么处理？", "acme", "public,it", "acme-device"),

    # ---------------- acme-devprocess ----------------
    ("一个迭代周期是多长？", "acme", "public,rd", "acme-devprocess"),
    ("核心模块的单元测试覆盖率要求多少？", "acme", "public,rd", "acme-devprocess"),
    ("迭代中途要插入新需求怎么办？", "acme", "public,rd", "acme-devprocess"),
    ("需求上线后要观察多久才能关闭？", "acme", "public,rd", "acme-devprocess"),
    ("需求评审多久开展一次？", "acme", "public,rd", "acme-devprocess"),

    # ---------------- acme-code ----------------
    ("主干分支叫什么名字？", "acme", "public,rd", "acme-code"),
    ("核心链路的代码要几个评审人通过？", "acme", "public,rd", "acme-code"),
    ("提交信息要带什么前缀？", "acme", "public,rd", "acme-code"),
    ("哪些内容禁止提交到代码库？", "acme", "public,rd", "acme-code"),
    ("静态扫描发现高危问题怎么处理？", "acme", "public,rd", "acme-code"),

    # ---------------- acme-release ----------------
    ("常规发布窗口安排在什么时间？", "acme", "public,rd", "acme-release"),
    ("灰度发布的顺序是怎样的？", "acme", "public,rd", "acme-release"),
    ("灰度期间错误率超过基线多少要回滚？", "acme", "public,rd", "acme-release"),
    ("回滚操作要在多久内完成？", "acme", "public,rd", "acme-release"),
    ("法定节假日前几天不安排发布？", "acme", "public,rd", "acme-release"),

    # ---------------- acme-dataclass ----------------
    ("数据一共分为几个级别？", "acme", "public,security", "acme-dataclass"),
    ("哪些数据属于绝密级？", "acme", "public,security", "acme-dataclass"),
    ("机密数据的访问日志保留多久？", "acme", "public,security", "acme-dataclass"),
    ("数据销毁要不要留存记录？", "acme", "public,security", "acme-dataclass"),
    ("员工离职时工作数据怎么处理？", "acme", "public,security", "acme-dataclass"),

    # ---------------- acme-incident ----------------
    ("发现安全事件多久内要通知值班人员？", "acme", "public,security", "acme-incident"),
    ("涉及个人信息泄露多久要报送监管？", "acme", "public,security", "acme-incident"),
    ("安全事件处置分为哪几个阶段？", "acme", "public,security", "acme-incident"),
    ("事件复盘报告多久内完成？", "acme", "public,security", "acme-incident"),
    ("什么情形直接判定为重大及以上事件？", "acme", "public,security", "acme-incident"),

    # ---------------- acme-contract ----------------
    ("什么样的合同需要法务审核？", "acme", "public,legal", "acme-contract"),
    ("合同金额超过一百万要谁审批？", "acme", "public,legal", "acme-contract"),
    ("用印内容和审批文本不一致怎么办？", "acme", "public,legal", "acme-contract"),
    ("合同签署后原件由谁保管？", "acme", "public,legal", "acme-contract"),
    ("法务审核的时限是多久？", "acme", "public,legal", "acme-contract"),

    # ---------------- acme-privacy ----------------
    ("收到删除个人信息的请求多久要响应？", "acme", "public,legal", "acme-privacy"),
    ("哪些属于敏感个人信息？", "acme", "public,legal", "acme-privacy"),
    ("个人信息出境要满足什么条件？", "acme", "public,legal", "acme-privacy"),
    ("用户拒绝提供非必要信息能拒绝服务吗？", "acme", "public,legal", "acme-privacy"),
    ("委托第三方处理个人信息要签什么？", "acme", "public,legal", "acme-privacy"),

    # ---------------- globex staff ----------------
    ("供应商资质不全多久内不能再申请？", "globex", "staff", "globex-supplier"),
    ("供应商的结算周期是多少天？", "globex", "staff", "globex-supplier"),
    ("交付延期十五天违约金怎么算？", "globex", "staff", "globex-supplier"),
    ("来料检验不合格多久内要退换？", "globex", "staff", "globex-supplier"),
    ("常温库的温度控制在什么范围？", "globex", "staff", "globex-warehouse"),
    ("同城订单的配送时效是多久？", "globex", "staff", "globex-warehouse"),
    ("盘点差异超过多少要上报？", "globex", "staff", "globex-warehouse"),
    ("在线客服首次响应不超过多久？", "globex", "staff", "globex-service"),
    ("重大投诉多久之内要上报？", "globex", "staff", "globex-service"),
    ("七天无理由退货从哪天开始算？", "globex", "staff", "globex-service"),
    ("A 级供应商的抽检比例是多少？", "globex", "staff", "globex-quality"),
    ("不合格品有哪几种处置方式？", "globex", "staff", "globex-quality"),
    ("报废金额超过多少要质量经理批？", "globex", "staff", "globex-quality"),
    ("控制点的检验记录要谁签字？", "globex", "staff", "globex-quality"),

    # ---------------- globex finance ----------------
    ("跨境贸易优先用什么币种结算？", "globex", "staff,finance", "globex-settlement"),
    ("超过五十万美元的付款要谁审批？", "globex", "staff,finance", "globex-settlement"),
    ("外币敞口超过多少要做套保？", "globex", "staff,finance", "globex-settlement"),
    ("跨境付款需要提供哪些单证？", "globex", "staff,finance", "globex-settlement"),
    ("库存商品采用什么方法计价？", "globex", "staff,finance", "globex-inventory"),
    ("存货成本包不包括仓储费用？", "globex", "staff,finance", "globex-inventory"),
    ("存货减值测试多久做一次？", "globex", "staff,finance", "globex-inventory"),
    ("库龄多久会被认定为呆滞库存？", "globex", "staff,finance", "globex-inventory"),

    # ====================================================================
    # 改写题：同一语义换说法，专门测语义召回（字面重合度低）
    # ====================================================================
    ("出差住酒店一晚上能报多少钱？", "acme", "public,finance", "acme-travel"),
    ("报销单据过期了还能补吗？", "acme", "public,finance", "acme-travel"),
    ("打车票还能单独拿去报账吗？", "acme", "public,finance", "acme-travel"),
    ("公司一年调几次薪水？", "acme", "public,hr", "acme-compensation"),
    ("年底能拿到多少奖金？", "acme", "public,hr", "acme-compensation"),
    ("P级一共划了多少档？", "acme", "public,hr", "acme-compensation"),
    ("绩效考核多久搞一次？", "acme", "public,hr", "acme-performance"),
    ("考核垫底会有什么后果？", "acme", "public,hr", "acme-performance"),
    ("想换个方向发展走什么路子？", "acme", "public,hr", "acme-training"),
    ("新人进来有师傅带吗？", "acme", "public,hr", "acme-training"),
    ("审批的人能不能同时有操作权？", "acme", "public,it", "acme-account"),
    ("服务器挂了多久要有人处理？", "acme", "public,it", "acme-ops"),
    ("改数据库结构有什么注意事项？", "acme", "public,it", "acme-ops"),
    ("代码合并之前要过几关？", "acme", "public,rd", "acme-code"),
    ("提交记录里不能出现什么东西？", "acme", "public,rd", "acme-code"),
    ("新功能上线出问题第一步做什么？", "acme", "public,rd", "acme-release"),
    ("客户资料算什么级别的数据？", "acme", "public,security", "acme-dataclass"),
    ("密钥材料归到哪一级？", "acme", "public,security", "acme-dataclass"),
    ("数据能随便发给合作方吗？", "acme", "public,security", "acme-dataclass"),
    ("被黑客攻击算几级事件？", "acme", "public,security", "acme-incident"),
    ("签合同要盖什么章？", "acme", "public,legal", "acme-contract"),
    ("合同条款改了要不要重新走审批？", "acme", "public,legal", "acme-contract"),
    ("收集用户信息之前要不要告知？", "acme", "public,legal", "acme-privacy"),
    ("刷脸的信息属于什么类别？", "acme", "public,legal", "acme-privacy"),
    ("想成为供应商要审什么？", "globex", "staff", "globex-supplier"),
    ("供应商老是延期交货会怎样？", "globex", "staff", "globex-supplier"),
    ("库房温度湿度有没有要求？", "globex", "staff", "globex-warehouse"),
    ("没有出库单能放行吗？", "globex", "staff", "globex-warehouse"),
    ("客户投诉多久要给答复？", "globex", "staff", "globex-service"),
    ("质量问题的退换货运费谁承担？", "globex", "staff", "globex-service"),
    ("来料抽检没过怎么办？", "globex", "staff", "globex-quality"),
    ("隔离的不合格品能混进合格区吗？", "globex", "staff", "globex-quality"),
    ("外币敞口怎么管？", "globex", "staff,finance", "globex-settlement"),
    ("反洗钱筛查是必须做的吗？", "globex", "staff,finance", "globex-settlement"),
    ("存货跌价准备怎么计提？", "globex", "staff,finance", "globex-inventory"),
    ("积压的库存怎么处置？", "globex", "staff,finance", "globex-inventory"),
    ("公积金缴纳基数怎么定？", "acme", "public", "acme-welfare"),
    ("员工结婚公司有礼金吗？", "acme", "public", "acme-welfare"),
    ("生孩子能休多长时间假？", "acme", "public", "acme-leave"),
    ("陪产假多久之内要用掉？", "acme", "public", "acme-leave"),
    ("上班迟到会被扣钱吗？", "acme", "public", "acme-leave"),
    ("周末加班能换成调休吗？", "acme", "public", "acme-leave"),
    ("会议室需要提前预定吗？", "acme", "public", "acme-conduct"),
    ("入职第一周要干些什么？", "acme", "public", "acme-onboarding"),
    ("试用期没通过会怎么样？", "acme", "public", "acme-onboarding"),
    ("预算不够了能先把钱花了吗？", "acme", "public,finance", "acme-budget"),
    ("比价要找几家供应商？", "acme", "public,finance", "acme-procurement"),
    ("电子发票可以报销吗？", "acme", "public,finance", "acme-invoice"),
    ("发票内容写得太笼统行不行？", "acme", "public,finance", "acme-invoice"),
    ("忘记电脑开机密码怎么办？", "acme", "public", "acme-se-baseline"),
    ("收到可疑链接应该怎么处理？", "acme", "public", "acme-se-baseline"),
    ("公司手册多久更新一次？", "acme", "public", "acme-handbook"),
    ("一年累计旷工多少天会被辞退？", "acme", "public", "acme-handbook"),
]


def main() -> None:
    out = []
    for i, (q, tenant, groups, gold) in enumerate(QA, 1):
        out.append({
            "qid": f"q{i:03d}",
            "question": q,
            "tenant_id": tenant,
            "groups": [g.strip() for g in groups.split(",")],
            "gold_doc": gold,
        })

    path = Path("eval/qa_set.json")
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    docs: dict[str, int] = {}
    tenants: dict[str, int] = {}
    for item in out:
        docs[item["gold_doc"]] = docs.get(item["gold_doc"], 0) + 1
        tenants[item["tenant_id"]] = tenants.get(item["tenant_id"], 0) + 1
    print(f"写入 {len(out)} 条 QA 到 {path}")
    print(f"覆盖文档 {len(docs)} 篇，租户分布 {tenants}")
    few = [d for d, n in docs.items() if n < 3]
    if few:
        print(f"题目少于 3 条的文档：{few}")


if __name__ == "__main__":
    main()
