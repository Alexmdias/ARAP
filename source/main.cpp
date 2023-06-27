#include <igl/opengl/glfw/Viewer.h>
#include <igl/opengl/glfw/imgui/ImGuiPlugin.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <igl/read_triangle_mesh.h>
#include <igl/cotmatrix.h>
#include <igl/adjacency_list.h>
#include <iostream>


Eigen::MatrixXd GetSi(Eigen::MatrixXd& V, Eigen::MatrixXd& newV, Eigen::SparseMatrix<double>& Wij, std::vector<std::vector<int>>& nList, int vi) 
{
    Eigen::SparseMatrix<double> diag(nList.at(vi).size(),nList.at(vi).size()); //diagonals are sparse
    auto P = Eigen::MatrixXd{V.cols(),nList.at(vi).size()}.setZero();
    auto newP = Eigen::MatrixXd{newV.cols(),nList.at(vi).size()}.setZero();
    int index = 0;
    for (int n:nList.at(vi))  //P contains "edges" as its columns (3 X Neigh(vi) matrix)
    {
      P.col(index) = V.row(vi) - V.row(n);
      newP.col(index) = newV.row(vi) - newV.row(n);//"""edges""" are e = pi - pn
      diag.coeffRef(index,index) = Wij.coeffRef(vi, n); //diagonal containing weights
      index++;
    }
    return P * diag * newP.transpose(); //Si = PDP'
}

void preCompute(Eigen::SparseMatrix<double>& L,std::map<int, Eigen::Vector3d>& controlMap, std::vector<std::vector<int>>& neighList,
  Eigen::SparseMatrix<double>& Wij, Eigen::SimplicialCholesky<Eigen::SparseMatrix<double>>& cholesky,Eigen::MatrixXd& b)
{
  //prepare L matrix for precomputation
  Eigen::SparseMatrix<double> Lcp = -L;
  
  for (auto const& c : controlMap) 
  {
    b.row(c.first) = c.second; //add constraints (fix control points)
    for(int i=0;i<Lcp.rows();i++)//make row and col 0 except for control
    {
      Lcp.coeffRef(c.first,i) = 0.0; 
      Lcp.coeffRef(i,c.first) = 0.0;
    }
    Lcp.coeffRef(c.first,c.first) = 1.0;
  }
  cholesky.compute(Lcp);
  //because we precomputed constraints on the right side, we need to update them also on the left side
  for (int i=0; i<b.rows(); i++)
    if(!controlMap.count(i)) //don't touch control points, if vertex is control point, stay fixed.
      for (int j:neighList.at(i))
        if (controlMap.count(j)) // Otherwise for every neighbor that is a control = Wij * controlpoint
          b.row(i) +=  Wij.coeffRef(i,j) * controlMap[j];
}

void ARAP(Eigen::MatrixXd& V, Eigen::MatrixXi& F,std::map<int, Eigen::Vector3d>&  controlMap, igl::opengl::glfw::Viewer& viewer)
{
  auto neighborsList = std::vector<std::vector<int>>(V.rows());
  igl::adjacency_list(F,neighborsList);
  Eigen::SparseMatrix<double> L(V.rows(),V.rows());
  igl::cotmatrix(V,F,L); 
  Eigen::SparseMatrix<double> Wij = L;
  for(int i=0;i<Wij.rows();i++) //cotangent weight's diagonal value are 0 because you can't be neighbor to yourself
    Wij.coeffRef(i,i)=0;

  Eigen::SimplicialCholesky<Eigen::SparseMatrix<double>>cholesky;
  auto b = Eigen::MatrixXd{V.rows(),V.cols()}.setZero();
  preCompute(L,controlMap,neighborsList,Wij,cholesky,b);

  auto newV = V;
  
  viewer.callback_pre_draw = [&](igl::opengl::glfw::Viewer & )->bool
  {
    if(!V.isApprox(newV,0.05) || V == newV) //stops animation when approx value is reached
    {
      std::vector<Eigen::MatrixXd> RList(V.rows()); //list of rotation matrices
      //local step
      for (int i = 0; i < V.rows(); i++)  
      {
        auto Si = GetSi(V, newV, Wij, neighborsList, i); //Si is covariance
        Eigen::JacobiSVD<Eigen::MatrixXd> decomp(Si, Eigen::ComputeThinU | Eigen::ComputeThinV); //fast for small matrices only
        RList[i] = decomp.matrixV() * decomp.matrixU().transpose(); // Ri = U*V
      }
      //global step
      auto newB = b;
      for (int i=0; i<V.rows(); i++)
        if(!controlMap.count(i)) //don't touch control points
          for (int j:neighborsList.at(i))  //summation for all neighbors of i = (wij/2)(Ri+Rj)(Pi-Pj)
            newB.row(i) += (Wij.coeffRef(i,j)/2.0) * (RList.at(i)+RList.at(j)) * (V.row(i)-V.row(j));
            
      V = newV;
      newV = cholesky.solve(newB);  //get new position. cholesky has been precomputed.
    }

    viewer.data().set_mesh(newV, F);
    viewer.core().align_camera_center(V);
    return false;
  };
  viewer.launch();
}


int main(int argc, char * argv[])
{
  int s = 0;
  while(true)
  {
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    igl::read_triangle_mesh("../../../../input/bunny.obj", V,F);

    double xyz[3];
    int index;
    std::map<int, Eigen::Vector3d>  controlMap;
    std::cout << "Select Option. Default : Do nothing \n";
    std::cout << "Quick demonstration: 1   Custom inputs: 2 \n";
    std::cin >> s;
    switch(s)
    {
      case 1:
       controlMap[1691] = Eigen::Vector3d {V.row(1202)[0]+0.1,V.row(1202)[1]+0.10,V.row(1202)[2]+0.01};//ear tip 
       controlMap[1600] = Eigen::Vector3d {V.row(1600)[0]-0.04,V.row(1600)[1]-0.03,V.row(1600)[2]-0.03};//ear tip 
       controlMap[1658] = Eigen::Vector3d {V.row(1658)[0]-0.04,V.row(1658)[1]-0.03,V.row(1658)[2]-0.03};//ear tip 
       controlMap[1868] = V.row(1868);//ear base
       controlMap[1365] = Eigen::Vector3d {V.row(1365)[0]-0.02,V.row(1365)[1]-0.004,V.row(1365)[2]+0.03}; //nose
       controlMap[2451] = V.row(2451);//paw
       controlMap[2199] = V.row(2199);//paw
       controlMap[2393] = V.row(2393);//butt
       controlMap[2413] = V.row(2413);//butt
       controlMap[955] = Eigen::Vector3d {V.row(955)[0]+0.1,V.row(955)[1]+0.1,V.row(955)[2]}; //tail
       break;
      case 2:
        do
        {
          std::cout << "\n\nChoose vertex index : ";
          std::cin >> index;
          std::cout << "Choose wanted x y z position \n";
          std::cout << "\nChoose x :";
          std::cin >> xyz[0];
          std::cout << "\nChoose y :";
          std::cin >> xyz[1];
          std::cout << "\nChoose z :";
          std::cin >> xyz[2];
          controlMap[index] = Eigen::Vector3d {xyz[0],xyz[1],xyz[2]};
          std::cout << "\nEnter 1 to add another control point, 0 to stop\n";
          std::cin >> s;
        }while(s==1);
        break;
      default:
        controlMap.clear();
    }
    //----------------render it----------------
    igl::opengl::glfw::Viewer viewer;
    igl::opengl::glfw::imgui::ImGuiPlugin plugin;
    igl::opengl::glfw::imgui::ImGuiMenu menu;
    plugin.widgets.push_back(&menu);
    viewer.plugins.push_back(&plugin);
    viewer.data().set_mesh(V, F);
    viewer.data().set_face_based(true);
    viewer.core().is_animating = true; 
    if(controlMap.size()>1)
      ARAP(V,F,controlMap,viewer);
    else
      viewer.launch();
  }
}
