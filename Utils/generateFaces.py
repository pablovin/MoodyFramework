from FaceEditing.Editor import editFace
from PIL import Image

ppoFace = "/home/pablo/Documents/Workspace/ChefsHatGYM/MoodyFramework/avatars/original/000010.jpg"
dqlFace = "/home/pablo/Documents/Workspace/ChefsHatGYM/MoodyFramework/avatars/original/000015.jpg"
randomFace ="/home/pablo/Documents/Workspace/ChefsHatGYM/MoodyFramework/avatars/original/000045.jpg"
a2cFace ="/home/pablo/Documents/Workspace/ChefsHatGYM/MoodyFramework/avatars/original/000104.jpg"


saveIn = "/home/pablo/Documents/Workspace/ChefsHatGYM/MoodyFramework/avatars"

agentFaces = [ppoFace, dqlFace, randomFace, a2cFace ]
agents = ["ppo", "dql", "random", "a2c"]

values = [[0.75,0.75], [0.75,-0.75], [0,0], [-0.75, 0.75], [-0.75,-0.75]]

for imgFile, agent in zip(agentFaces, agents):
    print ("Face:" + str(agent))
    for value in values:
        print ("Image:" + str(imgFile))
        face = editFace(imgFile, value[0], value[1])
        img = Image.fromarray(face.astype('uint8'))
        img = img.convert('RGB')
        img.save(saveIn + "/"+str(agent)+"a_"+ str(value[0]) + "_v_" + str(value[1]) + ".png")