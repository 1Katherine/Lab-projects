from lxml import etree
import os
import io
import argparse

maxRuntime=50

parser = argparse.ArgumentParser(description="shutDown")
parser.add_argument('-n', '--num', help='the num of port', type=str)
opts = parser.parse_args()

os.system('curl http://192.168.0.10:' + opts.num + '/jobs/ >> /usr/local/home/zwr/errorDetection/monitor.html')

html = io.open('/usr/local/home/zwr/errorDetection/monitor.html', encoding='UTF-8').read()

# 解析html
page = etree.HTML(html)


# 判断有无正在运行的任务，spark ui界面有kill按钮就表示该任务在运行
def ifActive():
    hrefs = page.xpath(u"/html/body/div[2]/div[4]/div/table/tbody/tr/td[2]/a[1]")
    for href in hrefs:
        return "kill" in href.text


# 获取job整体的执行时间
runTimes = page.xpath(u"/html/body/div[2]/div[2]/ul/li[2]/text()")
# 获取是否有执行失败的job
is_fail = page.xpath(u"/html/body/div[2]/div[2]/ul/li[4]/a/strong/text()")

# 没有正在运行的任务即退出运行
if not ifActive() and str(runTimes[1]) is None:
    print("未检测到有正在执行的spark程序")
    # 没有正在运行的任务
    exit(0)

# 运行时间只有秒级
if "s" in str(runTimes[1]):
    exit(0)

# 没有运行失败的job就退出
if len(is_fail) == 0:
    print("未检测到运行失败的任务")
    exit(0)
runTime = float(str(runTimes[1]).strip()[:-4])
# 检测到有运行失败的job或者任务的执行时间过长，则杀死该任务
if is_fail[0] == 'Failed Jobs:' or runTime > 15 or "h" in str(runTimes[1]):
    submitTimes = page.xpath(u"/html/body/div[2]/div[4]/div/table/tbody/tr/td[3]/text()")
    submitTimes = str(submitTimes[0]).strip()
    fname = '/usr/local/home/zwr/errorDetection/error.log'
    fw = open(fname, 'a')
    fw.write("Job Run Time:" + str(runTimes[1]).strip() + "\nJob Submit Time:" + submitTimes + "\n")
    kill_cmd = "/usr/local/home/zwr/errorDetection/kill.sh"
    os.system(kill_cmd)
    fw.write("Kill Successful" + "\n" + "-" * 50 + "\n")
