import MySQLdb
from objects import Objects
import json

def get_data():
    connection = MySQLdb.connect(host = "localhost", db = "jphacks", user = "jphacks", passwd = "jphacks")
    cur = connection.cursor()
    sql = "SELECT queues.id, structures.json FROM structures LEFT JOIN queues ON structures.queue_id = queues.id WHERE queues.state = 0"
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
    connection.commit()
    cur.close()
    connection.close()

def main():
    queue_id, decoded_json = get_data()
    objects = Objects(decoded_json)
    results = objects.calculate()
    put_data(queue_id, results)
    
if __name__ == "__main__":
    main()
