import HarfangHighLevel as hl

hl.Init(1024, 1024)

hl.AddFpsCamera(0,0,-0.3)

m = hl.Mat4
model = hl.Add3DFile("Avocado.fbx", m)

print(hl.GetRotation(m))


while not hl.UpdateDraw():
    pass

