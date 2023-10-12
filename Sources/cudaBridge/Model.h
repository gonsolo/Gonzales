// ======================================================================== //
// Copyright 2018-2019 Ingo Wald                                            //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#pragma once

#include "gdt/math/AffineSpace.h"
#include <vector>

/*! \namespace osc - Optix Siggraph Course */
namespace osc {
  using namespace gdt;
  
  /*! a simple indexed triangle mesh that our sample renderer will
      render */
  struct TriangleMesh {
    std::vector<vec3f> vertex;
    std::vector<vec3f> normal;
    std::vector<vec2f> texcoord;
    std::vector<vec3i> index;

    // material data:
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
}
