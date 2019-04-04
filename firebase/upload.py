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
    return module.absolute_save_path({"run id": run.id}) + module.ID + ".png"

def save_model_image(model, filepath):
    keras.utils.plot_model(model, to_file=filepath)

def upload_image(module, model=None, run_id = None):
    global run, db
    if not db: return

    if not run_id: run_id = run.id
    folder = module.absolute_save_path({'run id': run_id})
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, module.ID + ".png")
    if not module.model_image_path:
        return None
        save_model_image(model if model else module.keras_operation, filepath)
        module.model_image_path = filepath

    # Prepare for upload:
    client = storage.Client.from_service_account_json(service_account)
    bucket = client.get_bucket('ea-nas.appspot.com')
    blob = bucket.blob(folder + module.ID + ".png")
    blob.upload_from_filename(filename=filepath)
    return blob.path


def upload_population(modules):
    global db, run
    if not db: return

    batch = db.batch()
    population_docs = db.collection("runs/{}/population".format(run.id)).get()
    for doc in population_docs:
        batch.delete(doc.reference)

    for module in modules:
        if not module.model_image_link:
            module.model_image_link = upload_image(module)

        predecessor = module.predecessor.db_ref if module.predecessor else None
        modules_ref = db.collection("runs").document(run.id).collection("modules")
        ref = modules_ref.document(module.ID)
        batch.set(ref, {
            # Basic information:
            u'name': module.name,
            u'fitness': module.fitness,
            u'loss': module.loss,
            u'validationFitness': module.validation_fitness,
            u'validationLoss': module.validation_loss,
            u'evaluation': module.evaluation,
            u'epochs': len(module.fitness),
            u'numberOfOperations': module.number_of_operations(),
            u'predecessor': predecessor,

            # Database and visualization:
            u'modelImage': module.model_image_link,
            u'modelImageFileName': blob_filename(run, module),
            u'version': module.version,
            u'logs': module.logs,
        })
        pop_doc = db.collection("runs/{}/population".format(run.id)).document(module.ID)
        batch.set(pop_doc, {
            'module': 'runs/{}/modules/{}'.format(run.id, module.ID)
        })
        module.db_ref = 'runs/{}/modules/{}'.format(run.id, module.ID)
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
    config.started = datetime.datetime.now()
    timestamp, ref = db.collection("runs").add({
        u"dataset": config.dataset_name,
        u"epochsOfTraining": config.epochs_per_layer,
        u"batchSizeForTraining": config.batch_size,
        u"generations": config.generations,
        u"populationSize": config.population_size,
        u"started": config.started,
        u"status": "Running"
    })
    global run
    run = ref
    print("--> Created run {} in firebase!".format(run.id))
    return run.id

def update_run(config, status):
    global db, run
    if not db: return

    return db.collection("runs").document(run.id).set({
        u"dataset": config.dataset_name,
        u"epochsOfTraining": config.epochs,
        u"batchSizeForTraining": config.batch_size,
        u"generations": config.generations,
        u"populationSize": config.population_size,
        u"started": config.started,
        u"status": status
    })
