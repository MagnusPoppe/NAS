import datetime
import json
import os


def stop_firebase():
    global cred, app, db, run
    print("--> Firebase not in use.")
    cred = app = db = run = None

if "EA_NAS_UPLOAD_TO_FIREBASE" in os.environ and os.environ["EA_NAS_UPLOAD_TO_FIREBASE"] == "1":
    import firebase_admin
    from google.cloud import storage
    from firebase_admin import credentials
    from firebase_admin import firestore
    from tensorflow import keras

    # Login:
    service_account = './firebase/secrets/ea-nas-firebase-adminsdk-df1xu-be41ed0594.json'
    try:
        with open(service_account, "r") as file:
            secret = json.load(file)
        cred = credentials.Certificate(secret)
        app = firebase_admin.initialize_app(cred)
        db = firestore.client()
        run = None
        print("--> Firebase configured!")
    except:
        print("--> Firebase connection failed!")
        stop_firebase()
else:
    stop_firebase()


def blob_filename(run, module):
    return u"results/{}/{}/{}.png".format(run.id, module.name, module.version)


def upload_image(module):
    global run, db
    if not db: return

    folder = u"results/{}/{}/{}.png".format(run.id, module.name, module.version)
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, module.ID + ".png")
    if not module.model_image_path:
        keras.utils.plot_model(module.keras_operation, to_file=filepath)
        module.model_image_path = filepath

    # Prepare for upload:
    client = storage.Client.from_service_account_json(service_account)
    bucket = client.get_bucket('ea-nas.appspot.com')
    blob = bucket.blob(blob_filename(run, module))
    blob.upload_from_filename(filename=filepath)
    return blob.path


def upload_modules(modules):
    global db, run
    if not db: return

    batch = db.batch()
    population_docs = db.collection("runs/{}/population".format(run.id)).get()
    for doc in population_docs:
        batch.delete(doc.reference)

    for module in modules:
        module.model_image_link = upload_image(module)

        predecessor = module.predecessor.db_ref if module.predecessor else None
        modules_ref = db.collection("runs").document(run.id).collection("modules")
        ref = modules_ref.document(module.ID)
        batch.set(ref, {
            u'fitness': module.fitness,
            u'modelImage': module.model_image_link,
            u'modelImageFileName': blob_filename(run, module),
            u'epochs': module.epochs_trained,
            u'name': module.name,
            u'numberOfOperations': len(module.keras_operation.layers),
            u'version': module.version_number,
            u'predecessor': predecessor
        })

        for i, log in enumerate(module.logs):
            doc = ref.collection("logs").document()
            batch.set(doc, {
                u'index': i,
                u'value': log
            })
        pop_doc = db.collection("runs/{}/population".format(run.id)).document(module.ID)
        batch.set(pop_doc, {
            'module': 'runs/{}/modules/{}'.format(run.id, module.ID)
        })
        module.db_ref = 'runs/{}/modules/{}'.format(run.id, module.ID)
    batch.commit()

def update_fitness(modules):
    global db, run
    if not db: return

    batch = db.batch()
    for module in modules:
        predecessor = module.predecessor.db_ref if module.predecessor else None
        if module.db_ref:
            doc = db.collection(u"runs/{}/modules".format(run.id)).document(module.ID)
            batch.set(doc,  {
            u'fitness': module.fitness,
            u'modelImage': module.model_image_link,
            u'modelImageFileName': blob_filename(run, module),
            u'epochs': module.epochs_trained,
            u'name': module.name,
            u'numberOfOperations': len(module.keras_operation.layers),
            u'version': module.version_number,
            u'predecessor': predecessor
        })
    batch.commit()

def update_status(msg):
    global db, run
    if not db: return
    import builtins
    generation = builtins.generation
    db.collection("runs/{}/programOutput".format(run.id)).add({
        u'time': datetime.datetime.now(),
        u'generation': generation+1,
        u'msg': msg
    })


def create_new_run(config):
    global db
    if not db: return
    timestamp, ref = db.collection("runs").add({
        u"dataset": config['dataset'],
        u"epochsOfTraining": config['epochs'],
        u"batchSizeForTraining": config['batch size'],
        u"generations": config['generations'],
        u"populationSize": config['population size'],
        u"started": datetime.datetime.now()
    })
    global run
    run = ref
    print("--> Created run {} in firebase!".format(run.id))
    return run.id