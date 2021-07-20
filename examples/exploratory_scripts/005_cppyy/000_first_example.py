import cppyy

with open('test.c', 'r') as fid:
    code = fid.read()

# This works
cppyy.cppdef(code)
cppyy.gbl.p2m_rectmesh3d

# # This does not
# cppyy.cppexec(code)
# cppyy.gbl.p2m_rectmesh3d
