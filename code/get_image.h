#include <iostream>
#include <fstream>
#include <string>
 
 
class CImage
{
private:
  long  m_Width;
  long  m_Height;
     
public:
  CImage();
 ~CImage();
  int get_extension(std::string &fname);

  void LoadImage(std::string &fname);
   
  long get_width()
  {
    return m_Width;
  }
   
  long get_height()
  {
    return m_Height;
  } 
   
};
