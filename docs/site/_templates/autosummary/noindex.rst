{{ fullname | escape | underline }}

.. currentmodule:: {{ module }}

.. The object is indexed on the user-layer page (api/user.md); this re-export page is not, so
   Sphinx does not see two descriptions of one object (generate.py, api_tables).

{% if objtype == "class" -%}
.. autoclass:: {{ objname }}
   :members:
   :show-inheritance:
   :no-index:
{%- else -%}
.. auto{{ objtype }}:: {{ objname }}
   :no-index:
{%- endif %}
