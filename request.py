import grequests

urls = ['https://iz.ru/978363/2020-02-20/iandeks-vnov-stal-samoi-dorog..'] * 100

i = 0
while True:
  rs = [grequests.get(u) for u in urls]
  grequests.map(rs)
  i += len(urls)
  print(i)