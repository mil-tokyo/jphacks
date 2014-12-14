import time
import MySQLdb
from objects import Objects
import json

def get_data():
    connection = MySQLdb.connect(host = "localhost", db = "jphacks", user = "jphacks", passwd = "jphacks")
    cur = connection.cursor()
    sql = "SELECT queues.id, structures.json FROM structures LEFT JOIN queues ON structures.queue_id = queues.id WHERE queues.state = 0 ORDER BY structures.created_at ASC limit 1"
    results = cur.execute(sql)
    queue_id, rows = cur.fetchall()[0]
    cur.close()
    connection.close()
    decoded_json = json.loads(rows)
    return queue_id, decoded_json

def put_data(queue_id, data):
    encoded_json = json.dumps(data)
    print type(encoded_json)
    connection = MySQLdb.connect(host = "localhost", db = "jphacks", user = "jphacks", passwd = "jphacks")
    cur = connection.cursor()
    results = cur.execute("INSERT into results(queue_id, json, created_at) values(%s, %s, UNIX_TIMESTAMP())", (queue_id, encoded_json))
    results = cur.execute("UPDATE queues SET state=2 WHERE id = %s "%(queue_id))
    connection.commit()
    print "execute INSERT into results(queue_id, json, created_at) values(%s, %s, UNIX_TIMESTAMP())"%(queue_id, encoded_json)
    cur.close()
    connection.close()
    
def write_error(queue_id):
    connection = MySQLdb.connect(host = "localhost", db = "jphacks", user = "jphacks", passwd = "jphacks")
    cur = connection.cursor()
    results = cur.execute("UPDATE queues SET state=10 WHERE id = %s "%(queue_id))
    connection.commit()
    cur.close()
    connection.close()
    
def main():
    while 1:
        try:
            queue_id, decoded_json = get_data()
        except:
            time.sleep(0.5)
            continue
        try:
            objects = Objects(queue_id, decoded_json)
            results = objects.calculate()
        except:
            print "ERROR"
            write_error(queue_id)
            continue
        put_data(queue_id, results)
    
if __name__ == "__main__":
    main()
