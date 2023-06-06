from flask import Flask, request, jsonify, send_file, after_this_request
import json
import sys
import vdms
from subprocess import Popen
from collections import defaultdict
from math import log

app = Flask(__name__)
db = vdms.vdms()

# def start_vdms(port):
#     config = {
#         "port": port,
#         "db_root_path": "db",
#         "more-info": "github.com/IntelLabs/vdms"
#     }

#     config_file = '/tmp/config_'+str(port)+'.json'
#     with open(config_file, 'w') as fp:
#         json.dump(config, fp)

#     Popen("./../build/vdms -cfg " + config_file + " > screen.log 2> log.log &")

def setup_vdms_connection(host, port):
    db.connect(host, port)

def get_selection_score(response):

    entropy = 0
    count_dict = defaultdict(int)

    for r in response[0]["FindImage"]["entities"]:
        count_dict[r["label"]]+=1

    for key in count_dict.keys():
        p_x = count_dict[key]/float(response[0]["FindImage"]['returned'])
        print(key, p_x)

        if p_x > 0:
            entropy -= p_x*log(p_x)/log(2)

    print(entropy, len(count_dict), log(len(count_dict)), log(2))

    normalized_shannon_entropy = 0.0
    try:
        normalized_shannon_entropy = entropy/(log(len(count_dict))/log(2))
        print(normalized_shannon_entropy)
    except ZeroDivisionError:
        print("Single Class Label Only")
    
    return normalized_shannon_entropy

@app.route('/selection', methods=['POST'])
def selection():

    data = json.loads(request.data)
    print(data)

    find_response, res_arr = db.query([data])
    
    selection_score = get_selection_score(find_response)

    return(jsonify({"selection_score":selection_score}))

@app.route('/training', methods=['POST'])
def training():

    # data = json.loads(request.data)
    # print(data["FindImage"]["operations"][3]["options"])
    print(request)


    return(jsonify({"weights":""}))

if __name__ == '__main__':
    print(sys.argv)
    if len(sys.argv) != 3:
        print("Port missing\n Correct Usage: python3 client.py <port> <vdms_port>")
    else:
        # start_vdms(int(sys.argv[2]))
        setup_vdms_connection("localhost", int(sys.argv[2]))
        app.run(host='0.0.0.0', port=int(sys.argv[1]))
        