#pragma once

#include "gdt/math/AffineSpace.h"
#include <vector>

using namespace gdt;

struct TriangleMesh {
  std::vector<vec3f> vertex;
  std::vector<vec3f> normal;
  std::vector<vec2f> texcoord;
  std::vector<vec3i> index;

  vec3f              diffuse;
  int                diffuseTextureID { -1 };

      void addTriangle(float ax, float ay, float az, float bx, float by, float bz, float cx, float cy, float cz)
      {
          int firstVertexID = (int)vertex.size();
          affine3f xfm;
          xfm.p = vec3f(0.f, 0.f, 0.f);
          xfm.l.vx = vec3f(1.f ,0.f, 0.f);
          xfm.l.vy = vec3f(0.f, 1.f, 0.f);
          xfm.l.vz = vec3f(0.f, 0.f, 1.f);
          vertex.push_back(xfmPoint(xfm, vec3f( ax, ay, az)));
          vertex.push_back(xfmPoint(xfm, vec3f( bx, by, bz)));
          vertex.push_back(xfmPoint(xfm, vec3f( cx, cy, cz)));
      
          int indices[] = {0, 1, 2};
          index.push_back(firstVertexID+vec3i(indices[0],
                                              indices[1],
                                              indices[2]));
      }

};

struct Model {
  ~Model()
  {
    //for (auto mesh : meshes) delete mesh;
  }
  
  std::vector<TriangleMesh *> meshes;
};

Model *loadOBJ(const std::string &objFile);
