/*! \file data_collector.h
 *  \brief Contains a declaration of a data collector class that simpifies data collection when testing SkePU.
 */

#ifndef DATA_COLLECTOR_H
#define DATA_COLLECTOR_H

#include <string>
#include <set>
#include <utility>
#include <fstream>
#include <iomanip>

#include <iostream>

namespace skepu
{

enum DataExportFormat
{
   GNUPLOT,
   PREDICTION_FILE
};

/*!
 *  \ingroup testing
 */

/*!
 *  \class DataCollector2D
 *
 *  \brief A class that can be used to collect 2D data.
 *
 *  This class is used to simplify the collection of two dimensional data. It stores the data
 *  internally in a multiset of pairs. It can then be outputted in various formats. In the current
 *  version it only supports outputting to a file readable by GnuPlot.
 */
template <typename Tx, typename Ty>
class DataCollector2D
{

public:
   DataCollector2D(const std::string& _dataSetName, const std::string& _axisNameX, const std::string& _axisNameY);

   void addData(Tx x, Ty y);
   void clear();
   void writeDataToFile(const std::string& filename = "", DataExportFormat format = GNUPLOT);

   void appendDataToFile(const std::string& filename, bool firstTime, DataExportFormat format = GNUPLOT);

private:
   std::string dataSetName;
   std::pair<std::string, std::string> axisNames;
   std::multiset< std::pair<Tx, Ty> > dataSet;

   void writeGnuPlotFile(const std::string& filename);

   void appendGnuPlotFile(const std::string& filename, bool firstTime);

};

/*!
 *  The constructor sets some names for the current dataset.
 *
 *  \param _dataSetName Name of the dataset.
 *  \param _axisNameX Name of the X axis data, or rather the first data.
 *  \param _axisNameY Name of the Y axis data, or the second data.
 */
template <typename Tx, typename Ty>
DataCollector2D<Tx, Ty>::DataCollector2D(const std::string& _dataSetName, const std::string& _axisNameX, const std::string& _axisNameY)
{
   dataSetName = _dataSetName;
   axisNames.first = _axisNameX;
   axisNames.second = _axisNameY;
}

/*!
 *  Adds a 2D data point to the dataset.
 *
 *  \param x x coordinate.
 *  \param y y coordinate.
 */
template <typename Tx, typename Ty>
void DataCollector2D<Tx, Ty>::addData(Tx x, Ty y)
{
   dataSet.insert(std::make_pair(x, y));
}

/*!
 *  Clear the entire data set.
 */
template <typename Tx, typename Ty>
void DataCollector2D<Tx, Ty>::clear()
{
   dataSet.clear();
}

/*!
 *  Write the data to a file.
 *
 *  \param filename Filename of the data file to write to.
 *  \param format Format of the data. Currently only GNUPLOT.
 */
template <typename Tx, typename Ty>
void DataCollector2D<Tx, Ty>::writeDataToFile(const std::string& filename, DataExportFormat format)
{
   std::string _filename;

   if(filename.empty())
   {
      _filename = dataSetName + ".dat";
   }
   else
   {
      _filename = filename;
   }

   //Call right exporter
   if(format == GNUPLOT)
   {
      writeGnuPlotFile(_filename);
   }
}



/*!
 *  Append the data to a file.
 *
 *  \param filename Filename of the data file to appended to.
 *  \param format Format of the data. Currently only GNUPLOT.
 */
template <typename Tx, typename Ty>
void DataCollector2D<Tx, Ty>::appendDataToFile(const std::string& filename, bool firstTime, DataExportFormat format)
{
   std::string _filename;

   if(filename.empty())
   {
      _filename = dataSetName + ".dat";
   }
   else
   {
      _filename = filename;
   }

   //Call right exporter
   if(format == GNUPLOT)
   {
      appendGnuPlotFile(_filename, firstTime);
   }
}

/*!
 *  Write the data to a file readable by GnuPlot.
 *
 *  \param filename Filename of the target data file.
 */
template <typename Tx, typename Ty>
void DataCollector2D<Tx, Ty>::writeGnuPlotFile(const std::string& filename)
{
   std::ofstream file(filename.c_str());
   int tabLength = axisNames.first.length()+20;
   file<<std::left;
   file<<std::fixed <<std::setprecision(5);
   if(file.is_open())
   {
      //First add name and axis names as comments
      file<<"# " <<dataSetName <<"\n";
      file<<"# " <<std::setw(tabLength) <<axisNames.first <<std::setw(tabLength) <<axisNames.second <<"\n";

      //Add data in two columns
      for(typename std::multiset< std::pair<Tx, Ty> >::iterator it = dataSet.begin(); it != dataSet.end(); ++it)
      {
         file<<"  " <<std::setw(tabLength) <<it->first <<std::setw(tabLength) <<it->second <<"\n";
      }

      file.close();
   }
}



/*!
 *  Append the data to a file readable by GnuPlot.
 *
 *  \param filename Filename of the target data file.
 */
template <typename Tx, typename Ty>
void DataCollector2D<Tx, Ty>::appendGnuPlotFile(const std::string& filename, bool firstTime)
{
   std::ofstream file(filename.c_str(), std::ios::app);
   int tabLength = axisNames.first.length()+20;
   file<<std::left;
   file<<std::fixed <<std::setprecision(5);
   if(file.is_open())
   {
      if(firstTime)
      {
         //First add name and axis names as comments
         file<<"# " <<dataSetName <<"\n";
         file<<"# " <<std::setw(tabLength) <<axisNames.first <<std::setw(tabLength) <<axisNames.second <<"\n";
      }

      //Add data in two columns
      for(typename std::multiset< std::pair<Tx, Ty> >::iterator it = dataSet.begin(); it != dataSet.end(); ++it)
      {
         file<<"  " <<std::setw(tabLength) <<it->first <<std::setw(tabLength) <<it->second <<"\n";
      }

      file.close();
   }
}



}

#endif

