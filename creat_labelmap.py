
open('tf_files/label_map.pbtxt', 'w')

cats = open('data/Anno_fine/list_category_cloth.txt', 'r').readlines()
categories = list()
for i in cats[2:]:
	categories.append(i.split()[0])

end = '\n'
s = ' '
ID = 1

for name in categories:
	out = ''
	out += 'item' + s + '{'+end
	out += s*2 + 'id:' + s + str(ID) + end
	out += s*2 + 'name:' + s + '\'' + name + '\'' + end
	out += '}' + end*2
	
	with open('tf_files/label_map.pbtxt', 'a') as f:
		f.write(out)

	ID += 1

