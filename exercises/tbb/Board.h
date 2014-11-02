/*
    Copyright 2005-2014 Intel Corporation.  All Rights Reserved.

    This file is part of Threading Building Blocks. Threading Building Blocks is free software;
    you can redistribute it and/or modify it under the terms of the GNU General Public License
    version 2  as  published  by  the  Free Software Foundation.  Threading Building Blocks is
    distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
    implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
    See  the GNU General Public License for more details.   You should have received a copy of
    the  GNU General Public License along with Threading Building Blocks; if not, write to the
    Free Software Foundation, Inc.,  51 Franklin St,  Fifth Floor,  Boston,  MA 02110-1301 USA

    As a special exception,  you may use this file  as part of a free software library without
    restriction.  Specifically,  if other files instantiate templates  or use macros or inline
    functions from this file, or you compile this file and link it with other files to produce
    an executable,  this file does not by itself cause the resulting executable to be covered
    by the GNU General Public License. This exception does not however invalidate any other
    reasons why the executable file might be covered by the GNU General Public License.
*/

#ifndef __BOARD_H__
#define __BOARD_H__

#define WIN32_LEAN_AND_MEAN

#define LabelPtr int*
#define BoardPtr Board*

struct Matrix
{
    int width;
    int height;
    char* data;
};

class Board
    {
    public:
        Board(int width, int height, int squareSize, LabelPtr counter);
        virtual ~Board();
        void seed(int s);
        void seed(const BoardPtr s);
    public:
        Matrix *m_matrix;

    private:
        int m_width;
        int m_height;
        int m_squareSize;
        LabelPtr m_counter;
    };
#endif
