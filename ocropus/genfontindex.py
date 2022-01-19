import os
print("""
<html>
<head>
<title>Test</title>
</head>
<body>
<table>
""")

for i, fname in enumerate(os.popen("find . -name '*.ttf' | sort").readlines()):
    fname = fname.strip()
    print("""
        <tr>
        <td align='right'>
        %s&nbsp;&nbsp;&nbsp;
        </td>
        <td>
        <style>
        @font-face {
          font-family: 'Font%d';
          src: url('%s') format('truetype');
        }
        </style>
        <span style="font-family:Font%d;">0123456789 The quick brown fox jumped over the lazy dogs!</span><br>
        </td>
        </tr>
    """ % (fname, i, fname, i))

print("""
</table>
</body>
</html>
""")
