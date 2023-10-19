#pragma once

#include "gdt/math/AffineSpace.h"
#include <vector>

struct TriangleMesh {
  std::vector<gdt::vec3f> vertex;
  std::vector<gdt::vec3f> normal;
  std::vector<gdt::vec2f> texcoord;
  std::vector<gdt::vec3i> index;

  gdt::vec3f              diffuse;
  int                diffuseTextureID { -1 };

      void addTriangle(float ax, float ay, float az, float bx, float by, float bz, float cx, float cy, float cz)
      {
          int firstVertexID = (int)vertex.size();
	  gdt::affine3f xfm;
          xfm.p = gdt::vec3f(0.f, 0.f, 0.f);
          xfm.l.vx = gdt::vec3f(1.f ,0.f, 0.f);
          xfm.l.vy = gdt::vec3f(0.f, 1.f, 0.f);
          xfm.l.vz = gdt::vec3f(0.f, 0.f, 1.f);
          vertex.push_back(xfmPoint(xfm, gdt::vec3f( ax, ay, az)));
          vertex.push_back(xfmPoint(xfm, gdt::vec3f( bx, by, bz)));
          vertex.push_back(xfmPoint(xfm, gdt::vec3f( cx, cy, cz)));
      
          int indices[] = {0, 1, 2};
          index.push_back(firstVertexID+gdt::vec3i(indices[0], indices[1], indices[2]));
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
