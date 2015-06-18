output = open('simple_train_enrollment_feature.csv', 'w')
with open("simple_train.csv", 'r') as r:
    for line in r:
        items = line.strip().split(',')
        output.write('%s,%s' % (items[0], ','.join(items[3:])))
r.close()

output.close()

output = open('simple_test_enrollment_feature.csv', 'w')
with open("simple_test.csv", 'r') as r:
    for line in r:
        items =line.strip().split(',')
        output.write('%s,%s' % (items[0], ','.join(items[2:])))

output.close()
