import ntcore
import time
import threading

inst = 0
startThread = 0

def waitForConnect():
    """
    Waits for the robot to connect to the driver station.
    """
    while not isConnected():
        print("testing connection...")
        time.sleep(0.5)
    print("Connected to robot!")

def isConnected() -> bool:
    global inst
    """
    network tables takes like 15 secs to connect to the robot if the robot was already on when this code launched.\n
    if you just re-deploy code *after* launching this program then the time is only the time to initialize the robot code.\n
    so just run this program first then initialize robot to save time
    """
    DEFAULT_CHECK_STR = "this is not fms info" # This should be a string value that cannot be achieved on the robot and will be present on the robot at any given point
    # if the robot is connected to the driver station, since this wil be set to something else if connected properly.
    return inst.getTable("FMSInfo").getEntry(".type").getString(DEFAULT_CHECK_STR) != DEFAULT_CHECK_STR

def init():
    """
    Initializes the network tables client. This should be called before any other functions in this module.
    """
    global inst
    global startThread

    # Initialize ntcore on protocol 4
    inst = ntcore.NetworkTableInstance.getDefault()
    startThread = threading.Thread(target=waitForConnect)

    # start Network Tables Instance
    inst.startClient4("DS GUI Controller")
    inst.setServerTeam(3952)
    inst.startDSClient()

    startThread.start()

def getInstance() -> ntcore.NetworkTableInstance:
    global inst
    """
    Returns the network tables instance.
    """
    return inst

def getTable(tableName) -> ntcore.NetworkTable:
    global inst
    """
    Returns a table from the network tables server.
    """
    if not isConnected():
        print("Not connected to robot!")
    return inst.getTable(tableName)

def getEntry(tableName, entryName) -> ntcore.NetworkTableEntry:
    global inst
    """
    Returns an entry from a table from the network tables server.
    """
    if not isConnected():
        print("Not connected to robot!")
    return inst.getTable(tableName).getEntry(entryName)

init()